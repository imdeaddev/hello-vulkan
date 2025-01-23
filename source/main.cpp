#define GLM_ENABLE_EXPERIMENTAL
#include <optional>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
#include <numbers>
#include <string_view>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <stb_image.h>

constexpr std::string_view WINDOW_NAME = "Vulkan window";
constexpr int DEFAULT_WIDTH = 640;
constexpr int DEFAULT_HEIGHT = 480;

void glfw_size_callback(GLFWwindow *pWindow, int width, int height);

class RendererBase {
    GLFWwindow *m_pWindow = nullptr;
    friend void glfw_size_callback(GLFWwindow *pWindow, int width, int height);
    int m_width = 0;
    int m_height = 0;
    bool m_resized = false;

protected:
    bool is_size_changed_in_current_frame() const { return m_resized; }

public:
    virtual void init(int width, int height) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to init GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_pWindow = glfwCreateWindow(width, height, WINDOW_NAME.data(), nullptr, nullptr);
        if (!m_pWindow) {
            throw std::runtime_error("Failed to create window");
        }
        glfwSetWindowUserPointer(m_pWindow, this);
        glfwSetWindowSizeCallback(m_pWindow, glfw_size_callback);
    }

    VkResult create_surface_khr(VkInstance instance, VkSurfaceKHR *pOutSurface) const {
        if (!m_pWindow) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return glfwCreateWindowSurface(instance, m_pWindow, nullptr, pOutSurface);
    }

    std::vector<const char *> get_required_instance_extensions() const {
        uint32_t extensionsCount = 0;
        const char **ppExtensions = glfwGetRequiredInstanceExtensions(&extensionsCount);
        std::vector<const char *> requiredExtensions{extensionsCount};
        for (uint32_t i = 0; i < extensionsCount; ++i) {
            requiredExtensions[i] = ppExtensions[i];
        }
        return requiredExtensions;
    }

    bool is_running() { return m_pWindow && !glfwWindowShouldClose(m_pWindow); }

    virtual void update() {
        m_resized = false;
        glfwPollEvents();
    }

    virtual void destroy() {
        if (m_pWindow) {
            glfwDestroyWindow(m_pWindow);
        }
        glfwTerminate();
    }
};

void glfw_size_callback(GLFWwindow *pWindow, int width, int height) {
    auto pRendererBase = reinterpret_cast<RendererBase *>(glfwGetWindowUserPointer(pWindow));
    if (pRendererBase) {
        std::cerr << "Window resized to " << width << 'x' << height << std::endl;
        pRendererBase->m_width = width;
        pRendererBase->m_height = height;
        pRendererBase->m_resized = true;
    }
}

VkBool32 vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, unsigned int,
                               const VkDebugUtilsMessengerCallbackDataEXT *pMsgData, void *) {
    switch (severity) {

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        std::clog << "[V] ";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        std::clog << "[I] ";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        std::clog << "[W] ";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        std::clog << "[E] ";
        break;
    default:
        break;
    }
    std::clog << pMsgData->pMessage << std::endl;
    return VK_FALSE;
}

bool check_debug_utils_extension_support() {
    uint32_t propertyCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &propertyCount, nullptr);
    std::vector<VkExtensionProperties> allProperties{propertyCount};
    vkEnumerateInstanceExtensionProperties(nullptr, &propertyCount, allProperties.data());

    constexpr std::string_view DEBUG_UTILS_EXTENSION_NAME = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    return std::find_if(allProperties.begin(), allProperties.end(), [DEBUG_UTILS_EXTENSION_NAME](const VkExtensionProperties &properties) {
               return DEBUG_UTILS_EXTENSION_NAME == properties.extensionName;
           }) != allProperties.end();
}

VkResult create_debug_utils_messenger(VkInstance instance, VkDebugUtilsMessengerEXT *pOutMessenger) {
    auto createFunc =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (!createFunc) {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    createInfo.pfnUserCallback = vulkan_debug_callback;
    return createFunc(instance, &createInfo, nullptr, pOutMessenger);
}

void destroy_debug_utils_messenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
    auto destroyFunc =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroyFunc) {
        destroyFunc(instance, messenger, nullptr);
    }
}

struct Vertex {
    glm::vec3 position{};
    glm::vec2 texCoord{};
    glm::vec3 normal{};

    static const VkVertexInputBindingDescription binding_desc() {
        return VkVertexInputBindingDescription{.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    }

    static const std::array<VkVertexInputAttributeDescription, 2> attribute_descs() {
        std::array<VkVertexInputAttributeDescription, 2> descs{};
        descs[0].binding = 0;
        descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        descs[0].location = 0;
        descs[0].offset = offsetof(Vertex, position); // = 0;
        descs[1].binding = 0;
        descs[1].format = VK_FORMAT_R32G32_SFLOAT;
        descs[1].location = 1;
        descs[1].offset = offsetof(Vertex, texCoord); // = 3 * sizeof(float); OR = 12;
        return descs;
    }
};

struct ModelData {
    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};

    constexpr static ModelData plane() {
        ModelData result{};
        result.vertices = {
            {{1.0, -1.0, 0}, {1.0, 1.0}, {0.0, 0.0, -1.0}},  // Top-right
            {{1.0, 1.0, 0}, {1.0, 0.0}, {0.0, 0.0, -1.0}},   // Bottom-right
            {{-1.0, 1.0, 0}, {0.0, 0.0}, {0.0, 0.0, -1.0}},  // Bottom-left
            {{-1.0, -1.0, 0}, {0.0, 1.0}, {0.0, 0.0, -1.0}}, // Top-left
        };
        result.indices = {0, 1, 2, 3, 0, 2};
        return result;
    }
};

#define ThrowIfFailed(expr)                                                                                                                \
    do {                                                                                                                                   \
        if ((expr) != VK_SUCCESS) {                                                                                                        \
            throw std::runtime_error("Failed to execute expression: " #expr);                                                              \
        }                                                                                                                                  \
    } while (0)

struct QueueFamiliesProperties {
    std::optional<uint32_t> graphicsQueueFamily{};
    std::optional<uint32_t> presentQueueFamily{};

    bool is_complete() const { return graphicsQueueFamily.has_value() && presentQueueFamily.has_value(); }

    void load_queue_families(VkPhysicalDevice physicalDevice, VkSurfaceKHR surfaceKHR) {
        uint32_t queueFamilyPropertiesCount = 0;
        graphicsQueueFamily = std::nullopt;
        presentQueueFamily = std::nullopt;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProperties{queueFamilyPropertiesCount};
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

        uint32_t familyIndex = 0;
        for (auto &queueFamily : queueFamilyProperties) {
            if (!graphicsQueueFamily.has_value() && queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                --queueFamily.queueCount;
                graphicsQueueFamily = familyIndex;
            }
            VkBool32 presentSupported = VK_FALSE;
            if (!presentQueueFamily.has_value() && queueFamily.queueCount > 0 &&
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surfaceKHR, &presentSupported) == VK_SUCCESS &&
                presentSupported == VK_TRUE) {
                --queueFamily.queueCount;
                presentQueueFamily = familyIndex;
            }
            if (is_complete())
                return;
            ++familyIndex;
        }
    }
};

bool check_physical_device_extensions_support(VkPhysicalDevice device, const std::vector<const char *> &extensions) {
    uint32_t supportedExtensionsCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &supportedExtensionsCount, nullptr);
    std::vector<VkExtensionProperties> supportedExtensions{supportedExtensionsCount};
    vkEnumerateDeviceExtensionProperties(device, nullptr, &supportedExtensionsCount, supportedExtensions.data());

    for (std::string_view extension : extensions) {
        if (std::find_if(supportedExtensions.begin(), supportedExtensions.end(), [extension](const VkExtensionProperties &properties) {
                return extension == properties.extensionName;
            }) == supportedExtensions.end()) {
            return false;
        }
    }
    return true;
}

struct VulkanBuffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
};

struct SceneUniformBuffer {
    glm::mat4 view{};
    glm::mat4 proj{};
};

struct ObjectVSPushConstant {
    glm::mat4 model{};
};

struct ObjectFSPushConstant {
    glm::vec4 color{};
};

class VulkanRenderer : public RendererBase {
    static inline std::vector<const char *> debugLayers = {"VK_LAYER_KHRONOS_validation"};
    static inline std::vector<const char *> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    static constexpr uint32_t maxFramesInFlight = 3;
    static constexpr uint32_t preferredFramebuffersCount = 3;
    using base = RendererBase;

    // Instance-related objects
    VkInstance m_instance = nullptr;
    VkDebugUtilsMessengerEXT m_debugUtilsMessenger = nullptr;
    VkSurfaceKHR m_surfaceKHR = nullptr;

    // Physical device and device properties
    VkPhysicalDevice m_physicalDevice = nullptr;
    VkPhysicalDeviceProperties m_physicalDeviceProperties{};
    QueueFamiliesProperties m_queueFamilies{};
    std::vector<uint32_t> m_queueFamiliesForSwapchain{};

    // Back buffers properties
    VkSurfaceFormatKHR m_surfaceFormat{};
    VkFormat m_backBufferFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthBufferFormat = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR m_presentMode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t m_swapchainImagesCount = 0;
    VkExtent2D m_surfaceExtent{};
    VkSurfaceTransformFlagBitsKHR m_surfaceTransform{};

    // Logical device and queues
    VkDevice m_device = nullptr;
    VkQueue m_graphicsQueue = nullptr;
    VkQueue m_presentQueue = nullptr;

    // Graphics pipeline objects and main render pass
    VkDescriptorSetLayout m_sceneUniformBufferLayout = nullptr;
    VkPipelineLayout m_pipelineLayout = nullptr;
    VkPipeline m_pipeline = nullptr;
    VkRenderPass m_renderPass = nullptr;
    VkDescriptorPool m_descriptorPool = nullptr;
    std::vector<VkDescriptorSet> m_descriptorSets{};

    // Swapchain and back buffers
    VkSwapchainKHR m_swapchainKHR = nullptr;
    std::vector<VkImage> m_swapchainImages{};
    std::vector<VkImageView> m_swapchainImageViews{};
    std::vector<VkFramebuffer> m_framebuffers{};

    // Commmand buffers and synchronization objects
    VkCommandPool m_commandPool = nullptr;
    std::vector<VkCommandBuffer> m_commandBuffers{};
    // For CPU and GPU synchronization
    std::vector<VkFence> m_inFlightFences{};
    // For GPU queues synchronization
    std::vector<VkSemaphore> m_renderFinishedSemaphores{};
    std::vector<VkSemaphore> m_imageAvailableSemaphore{};

    std::vector<VulkanBuffer> m_sceneUniformBuffers{};
    std::vector<void *> m_sceneUniformMappings{};

    SceneUniformBuffer m_sceneData{};
    std::chrono::high_resolution_clock::time_point m_startTime{};
    std::chrono::high_resolution_clock::time_point m_lastTime{};

    VulkanBuffer m_vertexBuffer{};
    VulkanBuffer m_indexBuffer{};
    ObjectVSPushConstant m_objectTransform{};
    ObjectFSPushConstant m_objectMaterial{};
    uint32_t m_indexCount = 0;

    void create_instance_debug_utils_and_surface() {
        auto requiredExtensions = get_required_instance_extensions();
        bool debugUtilsSupported = check_debug_utils_extension_support();
        if (debugUtilsSupported) {
            requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;
        // other fields are optional

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        instanceInfo.enabledExtensionCount = requiredExtensions.size();
        instanceInfo.ppEnabledExtensionNames = requiredExtensions.data();
        if (debugUtilsSupported) {
            instanceInfo.enabledLayerCount = debugLayers.size();
            instanceInfo.ppEnabledLayerNames = debugLayers.data();
        }

        ThrowIfFailed(vkCreateInstance(&instanceInfo, nullptr, &m_instance));

        if (debugUtilsSupported) {
            if (create_debug_utils_messenger(m_instance, &m_debugUtilsMessenger) != VK_SUCCESS) {
                std::cerr << "Can't create debug utils messenger" << std::endl;
            }
        }

        ThrowIfFailed(create_surface_khr(m_instance, &m_surfaceKHR));
    }

    void choose_physical_device() {
        uint32_t physicalDevicesCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &physicalDevicesCount, nullptr);
        std::vector<VkPhysicalDevice> allDevices{physicalDevicesCount};
        vkEnumeratePhysicalDevices(m_instance, &physicalDevicesCount, allDevices.data());

        uint32_t maxScore = 0;
        for (auto deviceCandidate : allDevices) {
            QueueFamiliesProperties candidateQueues{};
            candidateQueues.load_queue_families(deviceCandidate, m_surfaceKHR);
            if (!candidateQueues.is_complete()) {
                continue;
            }
            if (!check_physical_device_extensions_support(deviceCandidate, requiredDeviceExtensions)) {
                continue;
            }
            VkPhysicalDeviceProperties candidateProperties{};
            vkGetPhysicalDeviceProperties(deviceCandidate, &candidateProperties);
            uint32_t candidateScore = 1;
            switch (candidateProperties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                candidateScore += 10000;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                candidateScore += 100;
                break;
            default:
                break;
            }
            if (candidateScore > maxScore) {
                maxScore = candidateScore;
                m_physicalDevice = deviceCandidate;
            }
        }
        if (m_physicalDevice == nullptr) {
            throw std::runtime_error("No suitable devices found");
        }
        m_queueFamilies.load_queue_families(m_physicalDevice, m_surfaceKHR);
        m_queueFamiliesForSwapchain.clear();
        m_queueFamiliesForSwapchain.push_back(m_queueFamilies.graphicsQueueFamily.value());
        m_queueFamiliesForSwapchain.push_back(m_queueFamilies.presentQueueFamily.value());
        m_queueFamiliesForSwapchain.erase(std::unique(m_queueFamiliesForSwapchain.begin(), m_queueFamiliesForSwapchain.end()));
        vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);
        std::clog << "GPU: " << m_physicalDeviceProperties.deviceName << std::endl;
    }

    VkFormat get_first_supported_format(const std::vector<VkFormat> &candidateFormats, VkImageTiling tiling,
                                        VkFormatFeatureFlags features = 0) const {
        for (auto format : candidateFormats) {
            VkFormatProperties formatProperties{};
            vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &formatProperties);
            if (tiling == VK_IMAGE_TILING_LINEAR && (formatProperties.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == VK_IMAGE_TILING_OPTIMAL && (formatProperties.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        return VK_FORMAT_UNDEFINED;
    }

    void get_surface_data() {
        VkSurfaceCapabilitiesKHR surfaceCapabilities{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surfaceKHR, &surfaceCapabilities);
        m_surfaceExtent = surfaceCapabilities.currentExtent;
        m_surfaceTransform = surfaceCapabilities.currentTransform;
        m_swapchainImagesCount =
            std::clamp<uint32_t>(preferredFramebuffersCount, surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount);

        uint32_t presentModesCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surfaceKHR, &presentModesCount, nullptr);
        std::vector<VkPresentModeKHR> allPresentModes{presentModesCount};
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surfaceKHR, &presentModesCount, allPresentModes.data());
        m_presentMode = allPresentModes[0];
        for (auto presentMode : allPresentModes) {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                m_presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }

        uint32_t formatsCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surfaceKHR, &formatsCount, nullptr);
        std::vector<VkSurfaceFormatKHR> surfaceFormats{formatsCount};
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surfaceKHR, &formatsCount, surfaceFormats.data());
        for (const auto &format : surfaceFormats) {
            if (format.format == VK_FORMAT_R8G8B8A8_SRGB) {
                m_surfaceFormat = format;
                break;
            }
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB) {
                m_surfaceFormat = format;
                break;
            }
            if (format.format == VK_FORMAT_R8G8B8A8_UNORM) {
                m_surfaceFormat = format;
                break;
            }
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM) {
                m_surfaceFormat = format;
                break;
            }
        }
        m_backBufferFormat = m_surfaceFormat.format;
        if (m_backBufferFormat == VK_FORMAT_UNDEFINED) {
            throw std::runtime_error("RGBA 32 bit formats for back buffer are not supported");
        }

        m_depthBufferFormat =
            get_first_supported_format({VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT},
                                       VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
        if (m_depthBufferFormat == VK_FORMAT_UNDEFINED) {
            throw std::runtime_error("Depth buffer with stencil is not supported");
        }
    }

    void update_surface_size_data() {
        VkSurfaceCapabilitiesKHR surfaceCapabilities{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surfaceKHR, &surfaceCapabilities);
        m_surfaceExtent = surfaceCapabilities.currentExtent;
        m_surfaceTransform = surfaceCapabilities.currentTransform;
        m_sceneData.proj =
            glm::perspectiveFov(90.f, static_cast<float>(m_surfaceExtent.width), static_cast<float>(m_surfaceExtent.height), 0.f, 1.f);
    }

    void create_logical_device() {
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> familiesToCountIndex{};
        familiesToCountIndex[m_queueFamilies.graphicsQueueFamily.value()] = std::make_pair(0, 0);
        familiesToCountIndex[m_queueFamilies.presentQueueFamily.value()] = std::make_pair(0, 0);
        ++familiesToCountIndex[m_queueFamilies.graphicsQueueFamily.value()].first;
        ++familiesToCountIndex[m_queueFamilies.presentQueueFamily.value()].first;

        std::vector<VkDeviceQueueCreateInfo> deviceQueuesInfos{};
        deviceQueuesInfos.reserve(familiesToCountIndex.size());
        constexpr std::array<float, 2> queuePriorities{1.0f, 1.0f};

        for (const auto [familyIndex, countIndex] : familiesToCountIndex) {
            const auto [queueCount, queueIndex] = countIndex;
            auto &queueInfo = deviceQueuesInfos.emplace_back();
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueCount = queueCount;
            queueInfo.pQueuePriorities = queuePriorities.data();
            queueInfo.queueFamilyIndex = familyIndex;
        }

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = deviceQueuesInfos.size();
        deviceInfo.pQueueCreateInfos = deviceQueuesInfos.data();
        deviceInfo.enabledExtensionCount = requiredDeviceExtensions.size();
        deviceInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();
        ThrowIfFailed(vkCreateDevice(m_physicalDevice, &deviceInfo, nullptr, &m_device));
        vkGetDeviceQueue(m_device, m_queueFamilies.graphicsQueueFamily.value(),
                         familiesToCountIndex[m_queueFamilies.graphicsQueueFamily.value()].second++, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, m_queueFamilies.presentQueueFamily.value(),
                         familiesToCountIndex[m_queueFamilies.presentQueueFamily.value()].second++, &m_presentQueue);
    }

    uint32_t find_memory_type_index(uint32_t mask, VkMemoryPropertyFlags flags) const {
        VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &deviceMemoryProperties);

        for (size_t i = 0; i < deviceMemoryProperties.memoryTypeCount; ++i) {
            if ((mask & 1 << i) && (deviceMemoryProperties.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type index");
    }

    void allocate_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags, VulkanBuffer &buffer) const {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.size = size;
        bufferInfo.usage = usage;

        ThrowIfFailed(vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer.buffer));

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, buffer.buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type_index(memoryRequirements.memoryTypeBits, memoryFlags);

        ThrowIfFailed(vkAllocateMemory(m_device, &allocInfo, nullptr, &buffer.memory));

        ThrowIfFailed(vkBindBufferMemory(m_device, buffer.buffer, buffer.memory, 0));
    }

    void destroy_buffer(const VulkanBuffer &buffer) const {
        if (buffer.buffer) {
            vkDestroyBuffer(m_device, buffer.buffer, nullptr);
        }
        if (buffer.memory) {
            vkFreeMemory(m_device, buffer.memory, nullptr);
        }
    }

    VkCommandBuffer begin_one_time_submit_buffer() const {
        VkCommandBuffer cmdBuffer{};
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        ThrowIfFailed(vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuffer));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        ThrowIfFailed(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
        return cmdBuffer;
    }

    void submit_one_time_submit_command_buffer(VkCommandBuffer cmdBuffer) const {
        vkEndCommandBuffer(cmdBuffer);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        ThrowIfFailed(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, nullptr));
        vkQueueWaitIdle(m_graphicsQueue);

        vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuffer);
    }

    void copy_buffer(VkCommandBuffer cmd, const VulkanBuffer &src, const VulkanBuffer &dst, VkDeviceSize size, VkDeviceSize srcOffset = 0,
                     VkDeviceSize dstOffset = 0) {
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &copyRegion);
    }

    void create_vertex_buffer(const std::vector<Vertex> &vertices, VulkanBuffer &buffer) {
        VulkanBuffer stagingBuffer{};
        VkDeviceSize bufferSize = vertices.size() * sizeof(vertices[0]);
        allocate_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);
        allocate_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer);
        void *pMapped = nullptr;
        ThrowIfFailed(vkMapMemory(m_device, stagingBuffer.memory, 0, bufferSize, 0, &pMapped));
        memcpy(pMapped, vertices.data(), bufferSize);
        vkUnmapMemory(m_device, stagingBuffer.memory);
        auto cmd = begin_one_time_submit_buffer();
        copy_buffer(cmd, stagingBuffer, buffer, bufferSize);
        submit_one_time_submit_command_buffer(cmd);
        destroy_buffer(stagingBuffer);
    }

    void create_index_buffer(const std::vector<uint32_t> &indices, VulkanBuffer &buffer) {
        VulkanBuffer stagingBuffer{};
        VkDeviceSize bufferSize = indices.size() * sizeof(indices[0]);
        allocate_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);
        allocate_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer);
        void *pMapped = nullptr;
        ThrowIfFailed(vkMapMemory(m_device, stagingBuffer.memory, 0, bufferSize, 0, &pMapped));
        memcpy(pMapped, indices.data(), bufferSize);
        vkUnmapMemory(m_device, stagingBuffer.memory);
        auto cmd = begin_one_time_submit_buffer();
        copy_buffer(cmd, stagingBuffer, buffer, bufferSize);
        submit_one_time_submit_command_buffer(cmd);
        destroy_buffer(stagingBuffer);
    }

    void load_model() {
        ModelData plane = ModelData::plane();
        create_vertex_buffer(plane.vertices, m_vertexBuffer);
        create_index_buffer(plane.indices, m_indexBuffer);
        m_indexCount = plane.indices.size();
    }

    void create_uniform_buffers() {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        ThrowIfFailed(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_sceneUniformBufferLayout));

        m_sceneUniformBuffers.resize(maxFramesInFlight);
        VkDeviceSize uboSize = sizeof(SceneUniformBuffer);
        for (auto &buffer : m_sceneUniformBuffers) {
            allocate_buffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer);
            void *pMapped = nullptr;
            ThrowIfFailed(vkMapMemory(m_device, buffer.memory, 0, uboSize, 0, &pMapped));
            m_sceneUniformMappings.push_back(pMapped);
            memcpy(pMapped, &m_sceneData, sizeof(SceneUniformBuffer));
        }

        VkDescriptorPoolSize poolSizes{};
        poolSizes.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.descriptorCount = maxFramesInFlight;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSizes;
        poolInfo.maxSets = maxFramesInFlight;

        ThrowIfFailed(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));

        std::vector<VkDescriptorSetLayout> layouts{maxFramesInFlight, m_sceneUniformBufferLayout};
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = m_descriptorPool;
        allocateInfo.descriptorSetCount = maxFramesInFlight;
        allocateInfo.pSetLayouts = layouts.data();
        m_descriptorSets.resize(maxFramesInFlight);
        ThrowIfFailed(vkAllocateDescriptorSets(m_device, &allocateInfo, m_descriptorSets.data()));

        for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
            update_descriptor_set(i);
        }
    }

    void update_descriptor_set(uint32_t i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_sceneUniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(SceneUniformBuffer);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_descriptorSets[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    }

    void create_pipeline_layout() {
        std::array<VkPushConstantRange, 2> objectPushConstants{};
        objectPushConstants[0].offset = 0;
        objectPushConstants[0].size = sizeof(ObjectVSPushConstant);
        objectPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        objectPushConstants[1].offset = sizeof(ObjectVSPushConstant);
        objectPushConstants[1].size = sizeof(ObjectFSPushConstant);
        objectPushConstants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_sceneUniformBufferLayout;
        pipelineLayoutInfo.pushConstantRangeCount = objectPushConstants.size();
        pipelineLayoutInfo.pPushConstantRanges = objectPushConstants.data();

        ThrowIfFailed(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
    }

    VkResult create_shader_from_path(const std::filesystem::path &path, VkShaderModule *pOutShader) const {
        std::ifstream ifs{path, std::ios::ate | std::ios::binary};
        if (!ifs.is_open()) {
            return VK_ERROR_UNKNOWN;
        }

        uint32_t shaderFileSize = ifs.tellg();
        std::vector<char> buffer;
        buffer.resize(shaderFileSize);
        ifs.seekg(0);
        ifs.read(buffer.data(), shaderFileSize);
        ifs.close();

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.pCode = reinterpret_cast<const uint32_t *>(buffer.data());
        shaderInfo.codeSize = shaderFileSize;

        return vkCreateShaderModule(m_device, &shaderInfo, nullptr, pOutShader);
    }

    void create_render_pass() {
        // Color attachment layout transitions: undefined -> color attachment optimal -> present src
        VkAttachmentDescription colorAttachmentDesc{};
        colorAttachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        colorAttachmentDesc.format = m_backBufferFormat;
        colorAttachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        VkAttachmentReference colorAttachment{};
        colorAttachment.attachment = 0;
        colorAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDependency subpassDependency{};
        subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        subpassDependency.dstSubpass = 0;
        subpassDependency.srcAccessMask = 0;
        subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubpassDescription subpassDesc{};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorAttachment;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDesc;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachmentDesc;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &subpassDependency;

        ThrowIfFailed(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass));
    }

    void create_pipeline() {
        VkShaderModule vertexShader{};
        VkShaderModule fragmentShader{};
        ThrowIfFailed(create_shader_from_path("triangle.vert.spv", &vertexShader));
        if (create_shader_from_path("triangle.frag.spv", &fragmentShader) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, vertexShader, nullptr);
            throw std::runtime_error("Failed to compile fragment shader");
        }

        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo pipelineDynamicStateInfo{};
        pipelineDynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        pipelineDynamicStateInfo.dynamicStateCount = dynamicStates.size();
        pipelineDynamicStateInfo.pDynamicStates = dynamicStates.data();

        VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateInfo{};
        pipelineVertexInputStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDesc = Vertex::binding_desc();
        auto attributeDesc = Vertex::attribute_descs();
        pipelineVertexInputStateInfo.vertexBindingDescriptionCount = 1;
        pipelineVertexInputStateInfo.pVertexBindingDescriptions = &bindingDesc;
        pipelineVertexInputStateInfo.vertexAttributeDescriptionCount = attributeDesc.size();
        pipelineVertexInputStateInfo.pVertexAttributeDescriptions = attributeDesc.data();

        VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateInfo{};
        pipelineInputAssemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        pipelineInputAssemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo pipelineViewportStateInfo{};
        pipelineViewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        pipelineViewportStateInfo.viewportCount = 1;
        pipelineViewportStateInfo.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateInfo{};
        pipelineRasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        pipelineRasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        pipelineRasterizationStateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
        pipelineRasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
        pipelineRasterizationStateInfo.lineWidth = 1.0f;
        pipelineRasterizationStateInfo.rasterizerDiscardEnable = false;

        VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateInfo{};
        pipelineMultisampleStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        pipelineMultisampleStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        pipelineMultisampleStateInfo.sampleShadingEnable = false;

        VkPipelineColorBlendAttachmentState pipelineColorBlendStateAttachment{};
        pipelineColorBlendStateAttachment.blendEnable = false;
        pipelineColorBlendStateAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateInfo{};
        pipelineColorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        pipelineColorBlendStateInfo.attachmentCount = 1;
        pipelineColorBlendStateInfo.pAttachments = &pipelineColorBlendStateAttachment;
        pipelineColorBlendStateInfo.logicOpEnable = false;

        std::array<VkPipelineShaderStageCreateInfo, 2> pipelineShaderStages{};
        auto &vertexShaderStage = pipelineShaderStages[0];
        vertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStage.module = vertexShader;
        vertexShaderStage.pName = "main";
        auto &fragmentShaderStage = pipelineShaderStages[1];
        fragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStage.module = fragmentShader;
        fragmentShaderStage.pName = "main";

        VkGraphicsPipelineCreateInfo graphicsPipelineInfo{};
        graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        graphicsPipelineInfo.pDynamicState = &pipelineDynamicStateInfo;
        graphicsPipelineInfo.pVertexInputState = &pipelineVertexInputStateInfo;
        graphicsPipelineInfo.pInputAssemblyState = &pipelineInputAssemblyStateInfo;
        graphicsPipelineInfo.pViewportState = &pipelineViewportStateInfo;
        graphicsPipelineInfo.pRasterizationState = &pipelineRasterizationStateInfo;
        graphicsPipelineInfo.pMultisampleState = &pipelineMultisampleStateInfo;
        graphicsPipelineInfo.pColorBlendState = &pipelineColorBlendStateInfo;
        graphicsPipelineInfo.stageCount = pipelineShaderStages.size();
        graphicsPipelineInfo.pStages = pipelineShaderStages.data();
        graphicsPipelineInfo.subpass = 0;
        graphicsPipelineInfo.renderPass = m_renderPass;
        graphicsPipelineInfo.layout = m_pipelineLayout;

        VkResult result = vkCreateGraphicsPipelines(m_device, nullptr, 1, &graphicsPipelineInfo, nullptr, &m_pipeline);
        vkDestroyShaderModule(m_device, vertexShader, nullptr);
        vkDestroyShaderModule(m_device, fragmentShader, nullptr);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline");
        }
    }

    void create_swapchain() {
        VkSwapchainCreateInfoKHR swapchainInfo{};
        swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainInfo.clipped = true;
        swapchainInfo.imageArrayLayers = 1;
        swapchainInfo.imageColorSpace = m_surfaceFormat.colorSpace;
        swapchainInfo.imageFormat = m_backBufferFormat;
        if (m_queueFamiliesForSwapchain.size() > 1) {
            swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapchainInfo.queueFamilyIndexCount = m_queueFamiliesForSwapchain.size();
            swapchainInfo.pQueueFamilyIndices = m_queueFamiliesForSwapchain.data();
        } else {
            swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            swapchainInfo.queueFamilyIndexCount = 0;
        }
        swapchainInfo.minImageCount = m_swapchainImagesCount;
        swapchainInfo.imageExtent = m_surfaceExtent;
        swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swapchainInfo.surface = m_surfaceKHR;
        swapchainInfo.presentMode = m_presentMode;
        swapchainInfo.preTransform = m_surfaceTransform;
        swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchainInfo.oldSwapchain = m_swapchainKHR;

        ThrowIfFailed(vkCreateSwapchainKHR(m_device, &swapchainInfo, nullptr, &m_swapchainKHR));

        if (swapchainInfo.oldSwapchain != nullptr) {
            vkDestroySwapchainKHR(m_device, swapchainInfo.oldSwapchain, nullptr);
        }

        vkGetSwapchainImagesKHR(m_device, m_swapchainKHR, &m_swapchainImagesCount, nullptr);
        m_swapchainImages.resize(m_swapchainImagesCount);
        vkGetSwapchainImagesKHR(m_device, m_swapchainKHR, &m_swapchainImagesCount, m_swapchainImages.data());
    }

    void create_swapchain_resources() {
        m_swapchainImageViews.resize(m_swapchainImagesCount);
        m_framebuffers.resize(m_swapchainImagesCount);

        for (uint32_t i = 0; i < m_swapchainImagesCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_swapchainImages[i];
            viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.format = m_backBufferFormat;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;

            ThrowIfFailed(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]));

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &m_swapchainImageViews[i];
            framebufferInfo.renderPass = m_renderPass;
            framebufferInfo.width = m_surfaceExtent.width;
            framebufferInfo.height = m_surfaceExtent.height;
            framebufferInfo.layers = 1;

            ThrowIfFailed(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]));
        }
    }

    void destroy_swapchain_resources() {
        for (auto framebuffer : m_framebuffers) {
            if (framebuffer) {
                vkDestroyFramebuffer(m_device, framebuffer, nullptr);
            }
        }
        for (auto view : m_swapchainImageViews) {
            if (view) {
                vkDestroyImageView(m_device, view, nullptr);
            }
        }
        m_swapchainImageViews.clear();
        m_framebuffers.clear();
        m_swapchainImages.clear();
    }

    void recreate_swapchain() {
        vkDeviceWaitIdle(m_device);
        update_surface_size_data();
        destroy_swapchain_resources();
        create_swapchain();
        create_swapchain_resources();
    }

    void create_command_buffers() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_queueFamilies.graphicsQueueFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        ThrowIfFailed(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = m_commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = maxFramesInFlight;

        m_commandBuffers.resize(maxFramesInFlight);

        ThrowIfFailed(vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, m_commandBuffers.data()));
    }

    void create_synchronization_objects() {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        m_inFlightFences.resize(maxFramesInFlight);
        m_imageAvailableSemaphore.resize(maxFramesInFlight);
        m_renderFinishedSemaphores.resize(maxFramesInFlight);

        for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
            ThrowIfFailed(vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]));
            ThrowIfFailed(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore[i]));
            ThrowIfFailed(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]));
        }
    }

public:
    void init(int width, int height) final {
        base::init(width, height);
        create_instance_debug_utils_and_surface();
        choose_physical_device();
        get_surface_data();
        create_logical_device();
        create_uniform_buffers();
        create_pipeline_layout();
        create_render_pass();
        create_pipeline();
        create_swapchain();
        create_swapchain_resources();
        create_command_buffers();
        create_synchronization_objects();
        load_model();

        m_sceneData.proj =
            glm::perspectiveFov(90.f, static_cast<float>(m_surfaceExtent.width), static_cast<float>(m_surfaceExtent.height), 0.f, 1.f);
        m_sceneData.view = glm::translate(glm::vec3{0.0, 0.0, -1.0});
        m_objectTransform.model = glm::mat4{1.0f};

        m_lastTime = std::chrono::high_resolution_clock::now();
        m_startTime = std::chrono::high_resolution_clock::now();

        std::clog << "Successfully created Vulkan renderer" << std::endl;
    }

    void update() final {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - m_startTime).count()) /
                     static_cast<float>(std::nano::den);
        float delta = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - m_lastTime).count()) /
                      static_cast<float>(std::nano::den);
        m_objectMaterial.color.r = (std::sin(time / std::numbers::pi * 2) + 1.0f) / 2.0f;
        m_objectMaterial.color.g = (std::cos(time / std::numbers::pi * 2) + 1.0f) / 2.0f;
        m_lastTime = currentTime;
        base::update();
        if (is_size_changed_in_current_frame()) {
            recreate_swapchain();
        }

        static uint32_t currentFrame = 0;

        vkWaitForFences(m_device, 1, &m_inFlightFences[currentFrame], true, std::numeric_limits<uint64_t>::max());
        vkResetFences(m_device, 1, &m_inFlightFences[currentFrame]);

        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(m_device, m_swapchainKHR, std::numeric_limits<uint64_t>::max(),
                                                m_imageAvailableSemaphore[currentFrame], nullptr, &imageIndex);
        switch (result) {
        case VK_ERROR_OUT_OF_DATE_KHR:
            recreate_swapchain();
            return;
        case VK_SUCCESS:
            [[fallthrough]];
        case VK_SUBOPTIMAL_KHR:
            break;
        default:
            throw std::runtime_error("Failed to acquire next swapchain image index");
        }

        const VkClearValue clearColor{0, 0, 0, 1};
        memcpy(m_sceneUniformMappings[currentFrame], &m_sceneData, sizeof(SceneUniformBuffer));
        VkCommandBuffer cmd = m_commandBuffers[currentFrame];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);
        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass = m_renderPass;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = &clearColor;
        renderPassBeginInfo.framebuffer = m_framebuffers[currentFrame];
        renderPassBeginInfo.renderArea.offset = {0, 0};
        renderPassBeginInfo.renderArea.extent = m_surfaceExtent;
        vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
            {
                VkViewport viewport{};
                viewport.x = 0.f;
                viewport.y = 0.f;
                viewport.width = static_cast<float>(m_surfaceExtent.width);
                viewport.height = static_cast<float>(m_surfaceExtent.height);
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                VkRect2D scissor{};
                scissor = renderPassBeginInfo.renderArea;
                vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSets[currentFrame], 0,
                                        nullptr);

                {
                    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ObjectVSPushConstant),
                                       &m_objectTransform);
                    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ObjectVSPushConstant),
                                       sizeof(ObjectFSPushConstant), &m_objectMaterial);
                    vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                    VkDeviceSize offsets[] = {0};
                    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, offsets);
                    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
                }
            }
        }
        vkCmdEndRenderPass(cmd);
        ThrowIfFailed(vkEndCommandBuffer(cmd));

        VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailableSemaphore[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinishedSemaphores[currentFrame];
        submitInfo.pWaitDstStageMask = &waitStages;

        ThrowIfFailed(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[currentFrame]));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[currentFrame];
        presentInfo.swapchainCount = 1;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pSwapchains = &m_swapchainKHR;

        result = vkQueuePresentKHR(m_presentQueue, &presentInfo);
        switch (result) {
        case VK_SUCCESS:
            break;
        case VK_SUBOPTIMAL_KHR:
            [[fallthrough]];
        case VK_ERROR_OUT_OF_DATE_KHR:
            recreate_swapchain();
            break;
        default:
            throw std::runtime_error("Failed to present image");
        }

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
    }

    void destroy() final {
        vkDeviceWaitIdle(m_device);

        destroy_buffer(m_vertexBuffer);
        destroy_buffer(m_indexBuffer);

        for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
            vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
            vkDestroySemaphore(m_device, m_imageAvailableSemaphore[i], nullptr);
            vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
        }

        if (m_commandPool) {
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }

        destroy_swapchain_resources();

        if (m_swapchainKHR) {
            vkDestroySwapchainKHR(m_device, m_swapchainKHR, nullptr);
        }

        if (m_pipeline) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }

        if (m_renderPass) {
            vkDestroyRenderPass(m_device, m_renderPass, nullptr);
        }

        for (auto &buffer : m_sceneUniformBuffers) {
            destroy_buffer(buffer);
        }
        m_sceneUniformBuffers.clear();
        m_sceneUniformMappings.clear();

        if (m_descriptorPool) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        }

        if (m_sceneUniformBufferLayout) {
            vkDestroyDescriptorSetLayout(m_device, m_sceneUniformBufferLayout, nullptr);
        }

        if (m_pipelineLayout) {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        }

        if (m_device) {
            vkDestroyDevice(m_device, nullptr);
        }

        if (m_surfaceKHR) {
            vkDestroySurfaceKHR(m_instance, m_surfaceKHR, nullptr);
        }

        if (m_debugUtilsMessenger) {
            destroy_debug_utils_messenger(m_instance, m_debugUtilsMessenger);
        }

        if (m_instance) {
            vkDestroyInstance(m_instance, nullptr);
        }
        std::clog << "Vulkan renderer destroyed" << std::endl;
        base::destroy();
    }
};

int main() {
    std::shared_ptr<RendererBase> renderer = std::make_shared<VulkanRenderer>();
    try {
        renderer->init(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        while (renderer->is_running()) {
            renderer->update();
        }
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
    }
    renderer->destroy();
    return 0;
}