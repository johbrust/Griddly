#include "Griddly/Core/GDY/Objects/Object.hpp"
#include "Griddly/Core/Grid.hpp"
#include "Griddly/Core/Observers/ASCIIObserver.hpp"
#include "Mocks/Griddly/Core/MockGrid.hpp"
#include "ObserverRTSTestData.hpp"
#include "ObserverTestData.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::AnyNumber;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Invoke;
using ::testing::Mock;
using ::testing::Pair;
using ::testing::Return;
using ::testing::ReturnRef;

namespace griddly {

void runASCIIObserverTest(ObserverConfig observerConfig,
                          Direction avatarDirection,
                          std::vector<uint32_t> expectedObservationShape,
                          std::vector<uint32_t> expectedObservationStride,
                          uint8_t* expectedData,
                          bool trackAvatar) {
  ObserverTestData testEnvironment = ObserverTestData(observerConfig, DiscreteOrientation(avatarDirection), trackAvatar);

  std::shared_ptr<ASCIIObserver> asciiObserver = std::shared_ptr<ASCIIObserver>(new ASCIIObserver(testEnvironment.mockGridPtr));

  asciiObserver->init(observerConfig);

  if (trackAvatar) {
    asciiObserver->setAvatar(testEnvironment.mockAvatarObjectPtr);
  }

  asciiObserver->reset();

  auto updateObservation = asciiObserver->update();

  ASSERT_EQ(asciiObserver->getTileSize(), glm::ivec2(1, 1));
  ASSERT_EQ(asciiObserver->getShape(), expectedObservationShape);
  ASSERT_EQ(asciiObserver->getStrides(), expectedObservationStride);

  size_t dataLength = asciiObserver->getShape()[0] * asciiObserver->getShape()[1] * asciiObserver->getShape()[2];

  auto updateObservationPointer = std::vector<uint8_t>(updateObservation, updateObservation + dataLength);

  ASSERT_THAT(updateObservationPointer, ElementsAreArray(expectedData, dataLength));

  testEnvironment.verifyAndClearExpectations();
}

void runASCIIObserverRTSTest(ObserverConfig observerConfig,
                             std::vector<uint32_t> expectedObservationShape,
                             std::vector<uint32_t> expectedObservationStride,
                             uint8_t* expectedData) {
  auto mockGridPtr = std::shared_ptr<MockGrid>(new MockGrid());

  ObserverRTSTestData testEnvironment = ObserverRTSTestData(observerConfig);

  std::shared_ptr<ASCIIObserver> asciiObserver = std::shared_ptr<ASCIIObserver>(new ASCIIObserver(testEnvironment.mockGridPtr));

  asciiObserver->init(observerConfig);

  asciiObserver->reset();

  auto updateObservation = asciiObserver->update();

  ASSERT_EQ(asciiObserver->getTileSize(), glm::ivec2(1, 1));
  ASSERT_EQ(asciiObserver->getShape(), expectedObservationShape);
  ASSERT_EQ(asciiObserver->getStrides()[0], expectedObservationStride[0]);
  ASSERT_EQ(asciiObserver->getStrides()[1], expectedObservationStride[1]);

  size_t dataLength = asciiObserver->getShape()[0] * asciiObserver->getShape()[1] * asciiObserver->getShape()[2];

  auto updateObservationPointer = std::vector<uint8_t>(updateObservation, updateObservation + dataLength);

  ASSERT_THAT(updateObservationPointer, ElementsAreArray(expectedData, dataLength));

  testEnvironment.verifyAndClearExpectations();
}

TEST(ASCIIObserverTest, defaultObserverConfig) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, partialObserver) {
  ObserverConfig config = {
      3,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][3][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 3, 5}, {1, 4, 12}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, partialObserver_withOffset) {
  ObserverConfig config = {
      3,
      5,
      0,
      1,
      false};

  uint8_t expectedData[5][3][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'.', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 3, 5}, {1, 4, 12}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar_rotateWithAvatar_NONE) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar_UP) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::UP, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}
TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar_RIGHT) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::RIGHT, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}
TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar_DOWN) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::DOWN, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}
TEST(ASCIIObserverTest, defaultObserverConfig_trackAvatar_LEFT) {
  ObserverConfig config = {
      5,
      5,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::LEFT, {4, 5, 5}, {1, 4, 20}, expectedData[0][0], false);
}

TEST(ASCIIObserverTest, partialObserver_trackAvatar_NONE) {
  ObserverConfig config = {
      5,
      3,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_trackAvatar_UP) {
  ObserverConfig config = {
      5,
      3,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::UP, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_trackAvatar_RIGHT) {
  ObserverConfig config = {
      5,
      3,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::RIGHT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_trackAvatar_DOWN) {
  ObserverConfig config = {
      5,
      3,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::DOWN, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}
TEST(ASCIIObserverTest, partialObserver_trackAvatar_LEFT) {
  ObserverConfig config = {
      5,
      3,
      0,
      0,
      false};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::LEFT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_NONE) {
  ObserverConfig config = {
      5,
      3,
      1,
      1,
      false};

  uint8_t expectedData[5][5][4] = {
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_UP) {
  ObserverConfig config = {
      5,
      3,
      1,
      1,
      false};

  uint8_t expectedData[5][5][4] = {
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::UP, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_RIGHT) {
  ObserverConfig config = {
      5,
      3,
      1,
      1,
      false};

  uint8_t expectedData[5][5][4] = {
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::RIGHT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_DOWN) {
  ObserverConfig config = {
      5,
      3,
      1,
      1,
      false};

  uint8_t expectedData[5][5][4] = {
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::DOWN, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}
TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_LEFT) {
  ObserverConfig config = {
      5,
      3,
      1,
      1,
      false};

  uint8_t expectedData[5][5][4] = {
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}},
      {{' ', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::LEFT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_rotateWithAvatar_NONE) {
  ObserverConfig config = {
      5,
      3,
      0,
      1,
      true};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::NONE, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_rotateWithAvatar_UP) {
  ObserverConfig config = {
      5,
      3,
      0,
      1,
      true};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::UP, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_rotateWithAvatar_RIGHT) {
  ObserverConfig config = {
      5,
      3,
      0,
      1,
      true};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::RIGHT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_rotateWithAvatar_DOWN) {
  ObserverConfig config = {
      5,
      3,
      0,
      1,
      true};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::DOWN, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, partialObserver_withOffset_trackAvatar_rotateWithAvatar_LEFT) {
  ObserverConfig config = {
      5,
      3,
      0,
      1,
      true};

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'Q', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'P', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'A', ' ', ' ', ' '}, {'.', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverTest(config, Direction::LEFT, {4, 5, 3}, {1, 4, 20}, expectedData[0][0], true);
}

TEST(ASCIIObserverTest, multiPlayer_Outline_Player1) {
  ObserverConfig config = {5, 5, 0, 0};
  config.playerId = 1;
  config.playerCount = 3;

  config.includePlayerId = true;

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '1', ' ', ' '}, {'B', '1', ' ', ' '}, {'C', '1', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '2', ' ', ' '}, {'B', '2', ' ', ' '}, {'C', '2', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '3', ' ', ' '}, {'B', '3', ' ', ' '}, {'C', '3', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverRTSTest(config, {4, 5, 5}, {1, 4, 4 * 5}, expectedData[0][0]);
}

TEST(ASCIIObserverTest, multiPlayer_Outline_Player2) {
  ObserverConfig config = {5, 5, 0, 0};
  config.playerId = 2;
  config.playerCount = 3;

  config.includePlayerId = true;

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '2', ' ', ' '}, {'B', '2', ' ', ' '}, {'C', '2', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '1', ' ', ' '}, {'B', '1', ' ', ' '}, {'C', '1', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '3', ' ', ' '}, {'B', '3', ' ', ' '}, {'C', '3', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverRTSTest(config, {4, 5, 5}, {1, 4, 4 * 5}, expectedData[0][0]);
}

TEST(ASCIIObserverTest, multiPlayer_Outline_Player3) {
  ObserverConfig config = {5, 5, 0, 0};
  config.playerId = 3;
  config.playerCount = 3;

  config.includePlayerId = true;

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '2', ' ', ' '}, {'B', '2', ' ', ' '}, {'C', '2', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '3', ' ', ' '}, {'B', '3', ' ', ' '}, {'C', '3', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '1', ' ', ' '}, {'B', '1', ' ', ' '}, {'C', '1', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverRTSTest(config, {4, 5, 5}, {1, 4, 4 * 5}, expectedData[0][0]);
}

TEST(ASCIIObserverTest, multiPlayer_Outline_Global) {
  ObserverConfig config = {5, 5, 0, 0};
  config.playerId = 0;
  config.playerCount = 3;

  config.includePlayerId = true;

  uint8_t expectedData[5][5][4] = {
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '1', ' ', ' '}, {'B', '1', ' ', ' '}, {'C', '1', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '2', ' ', ' '}, {'B', '2', ' ', ' '}, {'C', '2', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'A', '3', ' ', ' '}, {'B', '3', ' ', ' '}, {'C', '3', ' ', ' '}, {'W', ' ', ' ', ' '}},
      {{'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}, {'W', ' ', ' ', ' '}}};

  runASCIIObserverRTSTest(config, {4, 5, 5}, {1, 4, 4 * 5}, expectedData[0][0]);
}

}  // namespace griddly