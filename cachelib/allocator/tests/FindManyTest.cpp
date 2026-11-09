/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/Random.h>
#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "cachelib/allocator/CacheAllocator.h"
#include "cachelib/common/Time.h"

namespace facebook::cachelib::tests {
namespace {

using AllocatorT = LruAllocator;
using ReadHandle = AllocatorT::ReadHandle;
using Key = AllocatorT::Key;

constexpr uint32_t kValueSize = 100;

std::string keyOf(size_t i) { return fmt::format("key_{}", i); }

class FindManyTest : public testing::Test {
 protected:
  void SetUp() override {
    AllocatorT::Config config;
    config.setCacheSize(64 * Slab::kSize);
    // A small table keeps several items per bucket chain.
    config.setAccessConfig({10 /* bucketsPower */, 5 /* locksPower */});
    // The reaper would remove the item that Expired looks up.
    config.reaperInterval = std::chrono::seconds(0);
    cache_ = std::make_unique<AllocatorT>(config);
    pool_ =
        cache_->addPool("default", cache_->getCacheMemoryStats().ramCacheSize);
  }

  void insert(const std::string& key,
              uint32_t ttlSecs = 0,
              uint32_t creationTime = 0) {
    auto handle =
        cache_->allocate(pool_, key, kValueSize, ttlSecs, creationTime);
    ASSERT_NE(nullptr, handle);
    cache_->insertOrReplace(handle);
  }

  std::unique_ptr<AllocatorT> cache_;
  PoolId pool_{};
};

TEST_F(FindManyTest, MatchesFind) {
  constexpr size_t kItems = 2000;
  for (size_t i = 0; i < kItems; ++i) {
    insert(keyOf(i));
  }

  // Half hits, half misses, with duplicates; batch sizes around the internal
  // prefetch window of 32 keys and its sub-groups of 16.
  for (size_t batch : {0, 1, 5, 15, 16, 17, 31, 32, 33, 64, 100, 500}) {
    std::vector<std::string> storage;
    for (size_t i = 0; i < batch; ++i) {
      storage.push_back(keyOf(folly::Random::rand32(2 * kItems)));
    }
    std::vector<Key> keys(storage.begin(), storage.end());
    std::vector<ReadHandle> handles(batch);
    cache_->findMany(folly::range(keys), folly::range(handles));

    for (size_t i = 0; i < batch; ++i) {
      auto expected = cache_->find(keys[i]);
      ASSERT_EQ(expected.get(), handles[i].get())
          << "batch " << batch << ", key " << storage[i];
      if (handles[i]) {
        EXPECT_EQ(keys[i], handles[i]->getKey());
      }
    }
  }
}

TEST_F(FindManyTest, HoldsReferences) {
  insert("a");
  std::vector<Key> keys{"a", "a"};
  std::vector<ReadHandle> handles(2);
  cache_->findMany(folly::range(keys), folly::range(handles));
  ASSERT_NE(nullptr, handles[0]);
  EXPECT_EQ(2, handles[0]->getRefCount());
  handles.clear();
  EXPECT_EQ(1, cache_->find("a")->getRefCount());
}

TEST_F(FindManyTest, Stats) {
  insert("a");
  insert("b");
  const auto before = cache_->getGlobalCacheStats();
  std::vector<Key> keys{"a", "x", "b", "y", "a"};
  std::vector<ReadHandle> handles(keys.size());
  cache_->findMany(folly::range(keys), folly::range(handles));
  const auto after = cache_->getGlobalCacheStats();
  EXPECT_EQ(5, after.numCacheGets - before.numCacheGets);
  EXPECT_EQ(2, after.numCacheGetMiss - before.numCacheGetMiss);
}

TEST_F(FindManyTest, Expired) {
  insert("live");
  insert("expired", 1 /* ttlSecs */, util::getCurrentTimeSec() - 10);
  const auto before = cache_->getGlobalCacheStats();

  std::vector<Key> keys{"expired", "live"};
  std::vector<ReadHandle> handles(keys.size());
  cache_->findMany(folly::range(keys), folly::range(handles));
  EXPECT_EQ(nullptr, handles[0]);
  EXPECT_TRUE(handles[0].wasExpired());
  ASSERT_NE(nullptr, handles[1]);
  EXPECT_FALSE(handles[1].wasExpired());

  const auto after = cache_->getGlobalCacheStats();
  EXPECT_EQ(1, after.numCacheGetExpiries - before.numCacheGetExpiries);
  EXPECT_EQ(1, after.numCacheGetMiss - before.numCacheGetMiss);
}

TEST_F(FindManyTest, InvalidArguments) {
  insert("a");
  std::vector<Key> keys{"a", "b"};

  std::vector<ReadHandle> wrongSize(1);
  EXPECT_THROW(cache_->findMany(folly::range(keys), folly::range(wrongSize)),
               std::invalid_argument);

  std::vector<ReadHandle> notEmpty(2);
  notEmpty[1] = cache_->find("a");
  EXPECT_THROW(cache_->findMany(folly::range(keys), folly::range(notEmpty)),
               std::invalid_argument);
}

// Lookups race with inserts and removes in the same buckets; every handle
// returned must be for the requested key.
TEST_F(FindManyTest, ConcurrentMutations) {
  constexpr size_t kKeys = 4000;
  for (size_t i = 0; i < kKeys; i += 2) {
    insert(keyOf(i));
  }

  std::atomic<bool> stop{false};
  std::vector<std::thread> writers;
  for (int t = 0; t < 2; ++t) {
    writers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        const auto key = keyOf(folly::Random::rand32(kKeys));
        if (folly::Random::oneIn(2)) {
          if (auto handle = cache_->allocate(pool_, key, kValueSize)) {
            cache_->insertOrReplace(handle);
          }
        } else {
          cache_->remove(key);
        }
      }
    });
  }

  std::atomic<size_t> found{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&] {
      for (int iter = 0; iter < 2000; ++iter) {
        std::vector<std::string> storage;
        for (size_t i = 0; i < 40; ++i) {
          storage.push_back(keyOf(folly::Random::rand32(kKeys)));
        }
        std::vector<Key> keys(storage.begin(), storage.end());
        std::vector<ReadHandle> handles(keys.size());
        cache_->findMany(folly::range(keys), folly::range(handles));
        for (size_t i = 0; i < keys.size(); ++i) {
          if (handles[i]) {
            ASSERT_EQ(keys[i], handles[i]->getKey());
            found.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }
  for (auto& t : readers) {
    t.join();
  }
  stop = true;
  for (auto& t : writers) {
    t.join();
  }
  EXPECT_GT(found.load(), 0);
}

} // namespace
} // namespace facebook::cachelib::tests
