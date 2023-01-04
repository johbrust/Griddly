#pragma once

#include <map>
#include <string>
#include <vector>

#include "GDY/Actions/Direction.hpp"

namespace griddly {

class GameStateMapping {
 public:
  // global variable name -> global variable value
  std::map<std::string, uint32_t> globalVariableNameToIdx;

  // object name -> variable name -> variable value
  std::map<std::string, std::map<std::string, uint32_t>> objectVariableNameToIdx;
};

class GameObjectData {
 public:
  uint32_t id;
  std::string name;
  std::vector<int32_t> variables;

  // Some helper methods for deserializing
  inline const std::map<std::string, uint32_t>& getVariableIndexes(const GameStateMapping& gameStateMapping) const {
    return gameStateMapping.objectVariableNameToIdx.at(name);
  }

  inline const int32_t getVariableValue(const std::map<std::string, uint32_t>& objectVariableIndexes, const std::string& variableName) const {
    return variables[objectVariableIndexes.at(variableName)];
  }

  inline const void setVariableValue(const std::map<std::string, uint32_t>& objectVariableIndexes, const std::string& variableName, int32_t value) {
    variables[objectVariableIndexes.at(variableName)] = value;
  }

  inline const glm::ivec2 getLocation(const std::map<std::string, uint32_t>& objectVariableIndexes) const {
    return {getVariableValue(objectVariableIndexes, "_x"), getVariableValue(objectVariableIndexes, "_y")};
  }

  inline const DiscreteOrientation getOrientation(const std::map<std::string, uint32_t>& objectVariableIndexes) const {
    return DiscreteOrientation(
        glm::ivec2(
            getVariableValue(objectVariableIndexes, "_dx"),
            getVariableValue(objectVariableIndexes, "_dy")));
  }
};

class GridState {
 public:
  uint32_t width;
  uint32_t height;
};

class GameState {
 public:
  size_t hash;
  uint32_t playerCount;
  uint32_t tickCount;
  GridState grid;
  std::vector<uint32_t> defaultEmptyObjectIdx;
  std::vector<uint32_t> defaultBoundaryObjectIdx;
  std::vector<std::vector<int32_t>> globalData;
  std::vector<GameObjectData> objectData;
};

}  // namespace griddly