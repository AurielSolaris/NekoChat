// Vulkan compute backend. libvulkan is loaded at runtime so devices without a driver still start.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <unistd.h>

#include <vector>

#include "../common.h"
#include "compute.h"
#include "shaders_gen.h"

namespace neko::gpu {

namespace {

#define NEKO_VK_INSTANCE_FNS(X)                  \
    X(vkDestroyInstance)                         \
    X(vkEnumeratePhysicalDevices)                \
    X(vkGetPhysicalDeviceProperties)             \
    X(vkGetPhysicalDeviceQueueFamilyProperties)  \
    X(vkGetPhysicalDeviceMemoryProperties)       \
    X(vkEnumerateDeviceExtensionProperties)      \
    X(vkCreateDevice)                            \
    X(vkGetDeviceProcAddr)

#define NEKO_VK_DEVICE_FNS(X)          \
    X(vkDestroyDevice)                 \
    X(vkGetDeviceQueue)                \
    X(vkCreateBuffer)                  \
    X(vkDestroyBuffer)                 \
    X(vkGetBufferMemoryRequirements)   \
    X(vkAllocateMemory)                \
    X(vkFreeMemory)                    \
    X(vkBindBufferMemory)              \
    X(vkMapMemory)                     \
    X(vkUnmapMemory)                   \
    X(vkFlushMappedMemoryRanges)       \
    X(vkInvalidateMappedMemoryRanges)  \
    X(vkCreateShaderModule)            \
    X(vkDestroyShaderModule)           \
    X(vkCreateDescriptorSetLayout)     \
    X(vkDestroyDescriptorSetLayout)    \
    X(vkCreatePipelineLayout)          \
    X(vkDestroyPipelineLayout)         \
    X(vkCreateComputePipelines)        \
    X(vkDestroyPipeline)               \
    X(vkCreateDescriptorPool)          \
    X(vkDestroyDescriptorPool)         \
    X(vkResetDescriptorPool)           \
    X(vkAllocateDescriptorSets)        \
    X(vkUpdateDescriptorSets)          \
    X(vkCreateCommandPool)             \
    X(vkDestroyCommandPool)            \
    X(vkAllocateCommandBuffers)        \
    X(vkResetCommandBuffer)            \
    X(vkBeginCommandBuffer)            \
    X(vkEndCommandBuffer)              \
    X(vkCmdBindPipeline)               \
    X(vkCmdBindDescriptorSets)         \
    X(vkCmdPushConstants)              \
    X(vkCmdDispatch)                   \
    X(vkCmdPipelineBarrier)            \
    X(vkQueueSubmit)                   \
    X(vkCreateFence)                   \
    X(vkDestroyFence)                  \
    X(vkWaitForFences)                 \
    X(vkGetFenceStatus)                \
    X(vkResetFences)                   \
    X(vkDeviceWaitIdle)

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) fail(std::string("Vulkan: ") + what + " failed (" + std::to_string(int(r)) + ")");
}

struct VkBuf : Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* mapped = nullptr;
    bool coherent = true;
};

class VulkanBackend final : public ComputeBackend {
public:
    VulkanBackend() { init(); }
    ~VulkanBackend() override { destroy(); }

    const char* name() const override { return "Vulkan"; }
    std::string deviceName() const override { return deviceName_; }
    size_t maxBufferBytes() const override { return maxBuffer_; }

    Buffer* create(size_t bytes, const void* init, bool readback) override {
        auto b = std::make_unique<VkBuf>();
        b->size = bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = std::max<size_t>(bytes, 16);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bi, nullptr, &b->buffer), "vkCreateBuffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, b->buffer, &req);
        uint32_t type = pickMemoryType(req.memoryTypeBits, readback);
        b->coherent = (memProps_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        VkResult r = vkAllocateMemory(device_, &ai, nullptr, &b->memory);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(device_, b->buffer, nullptr);
            check(r, "vkAllocateMemory");
        }
        check(vkBindBufferMemory(device_, b->buffer, b->memory, 0), "vkBindBufferMemory");
        void* p = nullptr;
        check(vkMapMemory(device_, b->memory, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory");
        b->mapped = static_cast<uint8_t*>(p);
        if (init) upload(b.get(), 0, init, bytes);
        buffers_.push_back(std::move(b));
        return buffers_.back().get();
    }

    void upload(Buffer* buf, size_t offset, const void* src, size_t bytes) override {
        auto* b = static_cast<VkBuf*>(buf);
        std::memcpy(b->mapped + offset, src, bytes);
        if (!b->coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b->memory, 0, VK_WHOLE_SIZE};
            vkFlushMappedMemoryRanges(device_, 1, &range);
        }
    }

    void download(Buffer* buf, size_t offset, void* dst, size_t bytes) override {
        auto* b = static_cast<VkBuf*>(buf);
        if (!b->coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b->memory, 0, VK_WHOLE_SIZE};
            vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }
        std::memcpy(dst, b->mapped + offset, bytes);
    }

    // Descriptor sets live until the next begin(): every submission of the previous step has completed by then.
    void begin() override {
        check(vkResetDescriptorPool(device_, descPool_, 0), "vkResetDescriptorPool");
        startSlot();
    }

    void dispatch(Kernel k, Buffer* const bindings[4], const int32_t params[8], uint32_t gx, uint32_t gy) override {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &setLayout_;
        VkDescriptorSet set;
        check(vkAllocateDescriptorSets(device_, &ai, &set), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo infos[4];
        VkWriteDescriptorSet writes[4];
        for (int i = 0; i < 4; i++) {
            auto* b = static_cast<VkBuf*>(bindings[i] ? bindings[i] : dummy_);
            infos[i] = {b->buffer, 0, VK_WHOLE_SIZE};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = uint32_t(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
        VkCommandBuffer cmd = slots_[cur_].cmd;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[int(k)]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, params);
        vkCmdDispatch(cmd, gx, gy, 1);
        // Barriers order against everything earlier in submission order, so they also hold across flush().
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
    }

    // Submits the work so far and waits for it. Mali does not interleave queued work from different contexts,
    // so with anything of ours still queued the UI's and the keyboard's frames wait for the whole token; with
    // one layer in flight they get the GPU between layers.
    void flush() override {
        submitSlot();
        waitSlot(slots_[(cur_ + slots_.size() - 1) % slots_.size()]);
        startSlot();
    }

    void submitAndWait() override {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(slots_[cur_].cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                             &mb, 0, nullptr, 0, nullptr);
        submitSlot();
        for (Slot& s : slots_) waitSlot(s);
    }

private:
    // One step is recorded into a ring of command buffers; flush() submits the current one and moves on.
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool pending = false;
    };

    // Polls instead of blocking in vkWaitForFences: the UI renders through the same driver in this process,
    // and a thread parked inside the driver's wait can hold up the RenderThread's frames.
    void waitSlot(Slot& s) {
        if (!s.pending) return;
        VkResult r;
        while ((r = vkGetFenceStatus(device_, s.fence)) == VK_NOT_READY) usleep(200);
        check(r, "vkGetFenceStatus");
        check(vkResetFences(device_, 1, &s.fence), "vkResetFences");
        s.pending = false;
    }

    void startSlot() {
        Slot& s = slots_[cur_];
        waitSlot(s);  // only when a step needs more submissions than the ring holds
        check(vkResetCommandBuffer(s.cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(s.cmd, &bi), "vkBeginCommandBuffer");
    }

    void submitSlot() {
        Slot& s = slots_[cur_];
        check(vkEndCommandBuffer(s.cmd), "vkEndCommandBuffer");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s.cmd;
        check(vkQueueSubmit(queue_, 1, &si, s.fence), "vkQueueSubmit");
        s.pending = true;
        cur_ = (cur_ + 1) % slots_.size();
    }

    bool hasDeviceExtension(const char* name) {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
        for (auto& e : exts)
            if (std::strcmp(e.extensionName, name) == 0) return true;
        return false;
    }

    void init() {
        lib_ = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) fail("libvulkan.so not available");
        vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib_, "vkGetInstanceProcAddr"));
        if (!vkGetInstanceProcAddr) fail("vkGetInstanceProcAddr missing");
        auto vkCreateInstance =
            reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
        if (!vkCreateInstance) fail("vkCreateInstance missing");

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "NekoChat";
        app.pEngineName = "NekoEngine";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
#define X(fn) fn = reinterpret_cast<PFN_##fn>(vkGetInstanceProcAddr(instance_, #fn)); if (!fn) fail("missing " #fn);
        NEKO_VK_INSTANCE_FNS(X)
#undef X

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) fail("no Vulkan physical device");
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(instance_, &count, devs.data());
        for (auto d : devs) {
            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qs(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qs.data());
            for (uint32_t i = 0; i < qn; i++) {
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    phys_ = d;
                    queueFamily_ = i;
                    break;
                }
            }
            if (phys_) break;
        }
        if (!phys_) fail("no Vulkan compute queue");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_, &props);
        deviceName_ = props.deviceName;
        maxBuffer_ = props.limits.maxStorageBufferRange;
        if (props.limits.maxComputeWorkGroupCount[0] < 65535 || props.limits.maxComputeSharedMemorySize < 8192)
            fail("Vulkan device limits too small");
        vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        // A low-priority queue (where supported) lets the UI's and the keyboard's GPU work go first.
        VkDeviceQueueGlobalPriorityCreateInfoEXT prioInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT};
        prioInfo.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW_EXT;
        const char* prioExt = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
        VkResult created = VK_ERROR_INITIALIZATION_FAILED;
        if (hasDeviceExtension(prioExt)) {
            qci.pNext = &prioInfo;
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = &prioExt;
            created = vkCreateDevice(phys_, &dci, nullptr, &device_);
            lowPriority_ = created == VK_SUCCESS;
        }
        if (created != VK_SUCCESS) {
            qci.pNext = nullptr;
            dci.enabledExtensionCount = 0;
            dci.ppEnabledExtensionNames = nullptr;
            check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
        }
#define X(fn) fn = reinterpret_cast<PFN_##fn>(vkGetDeviceProcAddr(device_, #fn)); if (!fn) fail("missing " #fn);
        NEKO_VK_DEVICE_FNS(X)
#undef X
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

        VkDescriptorSetLayoutBinding binds[4];
        for (uint32_t i = 0; i < 4; i++)
            binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 4;
        lci.pBindings = binds;
        check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &setLayout_), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device_, &pli, nullptr, &pipeLayout_), "vkCreatePipelineLayout");

        for (int k = 0; k < int(Kernel::Count); k++) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = kShaderSpvSize[k];
            smi.pCode = reinterpret_cast<const uint32_t*>(kShaderSpv[k]);
            VkShaderModule mod;
            check(vkCreateShaderModule(device_, &smi, nullptr, &mod), "vkCreateShaderModule");
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = mod;
            cpi.stage.pName = "main";
            cpi.layout = pipeLayout_;
            VkResult r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipelines_[k]);
            vkDestroyShaderModule(device_, mod, nullptr);
            check(r, kShaderNames[k]);
        }

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSets * 4};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = kMaxSets;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(device_, &dpi, nullptr, &descPool_), "vkCreateDescriptorPool");

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queueFamily_;
        check(vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_), "vkCreateCommandPool");
        slots_.resize(kSlots);
        for (Slot& s : slots_) {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = cmdPool_;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device_, &cai, &s.cmd), "vkAllocateCommandBuffers");
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device_, &fci, nullptr, &s.fence), "vkCreateFence");
        }

        dummy_ = create(16, nullptr, false);
        NEKO_LOGI("Vulkan backend on %s (max storage buffer %zu MB, %s priority queue)", deviceName_.c_str(),
                  maxBuffer_ >> 20, lowPriority_ ? "low" : "default");
    }

    uint32_t pickMemoryType(uint32_t allowed, bool readback) const {
        int best = -1, bestScore = -1;
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++) {
            if (!(allowed & (1u << i))) continue;
            VkMemoryPropertyFlags f = memProps_.memoryTypes[i].propertyFlags;
            if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            int score = 0;
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += readback ? 1 : 4;
            if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 2;
            if (readback && (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) score += 8;
            if (score > bestScore) { bestScore = score; best = int(i); }
        }
        if (best < 0) fail("Vulkan: no host visible memory type");
        return uint32_t(best);
    }

    void destroy() {
        if (device_) {
            vkDeviceWaitIdle(device_);
            for (auto& b : buffers_) {
                vkUnmapMemory(device_, b->memory);
                vkDestroyBuffer(device_, b->buffer, nullptr);
                vkFreeMemory(device_, b->memory, nullptr);
            }
            buffers_.clear();
            for (Slot& s : slots_)
                if (s.fence) vkDestroyFence(device_, s.fence, nullptr);
            if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
            if (descPool_) vkDestroyDescriptorPool(device_, descPool_, nullptr);
            for (auto p : pipelines_)
                if (p) vkDestroyPipeline(device_, p, nullptr);
            if (pipeLayout_) vkDestroyPipelineLayout(device_, pipeLayout_, nullptr);
            if (setLayout_) vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ && vkDestroyInstance) vkDestroyInstance(instance_, nullptr);
        if (lib_) dlclose(lib_);
    }

    static constexpr uint32_t kMaxSets = 1024;
    static constexpr size_t kSlots = 32;

    void* lib_ = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define X(fn) PFN_##fn fn = nullptr;
    NEKO_VK_INSTANCE_FNS(X)
    NEKO_VK_DEVICE_FNS(X)
#undef X
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkPhysicalDeviceMemoryProperties memProps_{};
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipelines_[int(Kernel::Count)] = {};
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    std::vector<Slot> slots_;
    size_t cur_ = 0;
    bool lowPriority_ = false;
    std::vector<std::unique_ptr<VkBuf>> buffers_;
    Buffer* dummy_ = nullptr;
    std::string deviceName_;
    size_t maxBuffer_ = 0;
};

}  // namespace

std::unique_ptr<ComputeBackend> createVulkan(std::string& error) {
    try {
        return std::make_unique<VulkanBackend>();
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

}  // namespace neko::gpu
