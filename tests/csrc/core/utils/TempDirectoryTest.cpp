#include <gtest/gtest.h>

#include <filesystem>

#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

using namespace at::neuron;

TEST(TempDirectoryTest, Prefix_PreserveFalse_DirectoryCleanedUp) {
  std::string path;
  {
    TempDirectory temp("test_cleanup_", false, false);
    path = temp.path();
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(TempDirectoryTest, Prefix_PreserveTrue_DirectoryPersists) {
  std::string path;
  {
    TempDirectory temp("test_preserve_", false, true);
    path = temp.path();
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove_all(path);
}

TEST(TempDirectoryTest, FullPath_PreserveFalse_DirectoryCleanedUp) {
  std::string path = "/tmp/test_fullpath_cleanup";
  {
    TempDirectory temp(path, true, false);
    EXPECT_EQ(temp.path(), path);
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(TempDirectoryTest, FullPath_PreserveTrue_DirectoryPersists) {
  std::string path = "/tmp/test_fullpath_preserve";
  {
    TempDirectory temp(path, true, true);
    EXPECT_EQ(temp.path(), path);
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove_all(path);
}
