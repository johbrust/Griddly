#pragma once
#include <glm/glm.hpp>
#include <memory>

#include "Vulkan/VulkanObserver.hpp"

namespace griddly {

class VulkanGridObserver : public VulkanObserver {
 public:
  VulkanGridObserver(std::shared_ptr<Grid> grid, ResourceConfig resourceConfig, ShaderVariableConfig shaderVariableConfig);
  ~VulkanGridObserver();

  void init(ObserverConfig observerConfig) override;

 protected:
  virtual void renderLocation(glm::ivec2 objectLocation, glm::ivec2 outputLocation, glm::ivec2 tileOffset, DiscreteOrientation renderOrientation) const = 0;

  virtual void updateCommandBuffers(uint32_t numObjects) const;
  virtual void updatePersistentShaderBuffers();
  virtual uint32_t updateFrameShaderBuffers() = 0;

  void resetShape() override;
  virtual std::vector<VkRect2D> calculateDirtyRectangles(std::unordered_set<glm::ivec2> updatedLocations) const;
  
};

}  // namespace griddly