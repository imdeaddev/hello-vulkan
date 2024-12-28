#include <optional>
#include <vector>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string_view>

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

class VulkanRenderer : public RendererBase {
    static inline std::vector<const char *> debugLayers = {"VK_LAYER_KHRONOS_validation"};
    static inline std::vector<const char *> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    using base = RendererBase;
    VkInstance m_instance = nullptr;
    VkDebugUtilsMessengerEXT m_debugUtilsMessenger = nullptr;
    VkSurfaceKHR m_surfaceKHR = nullptr;

    VkPhysicalDevice m_physicalDevice = nullptr;
    VkPhysicalDeviceProperties m_physicalDeviceProperties{};
    QueueFamiliesProperties m_queueFamilies{};

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
        vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);
        std::clog << "GPU: " << m_physicalDeviceProperties.deviceName << std::endl;
    }

public:
    void init(int width, int height) final {
        base::init(width, height);
        create_instance_debug_utils_and_surface();
        choose_physical_device();

        std::clog << "Successfully created Vulkan renderer" << std::endl;
    }

    void update() final {
        base::update();
        if (is_size_changed_in_current_frame()) {
            // TODO: recreate swapchain and image resources
        }
        // TODO: draw
    }

    void destroy() final {
        if (m_surfaceKHR) {
            vkDestroySurfaceKHR(m_instance, m_surfaceKHR, nullptr);
        }

        if (m_debugUtilsMessenger) {
            destroy_debug_utils_messenger(m_instance, m_debugUtilsMessenger);
        }

        if (m_instance) {
            vkDestroyInstance(m_instance, nullptr);
        }
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