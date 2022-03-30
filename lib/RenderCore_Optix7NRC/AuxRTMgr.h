#pragma once

#include "core_settings.h"
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include "shared_host_code/cudatools.h"

// This class uses the SOTA Surface Object API
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
class AuxRTMgr {
private:
    struct AuxRT {
        // this means the upper code may chose not to update this texture
        bool interested;
        bool linked;
        bool bound;
        GLTexture *texture;
        cudaGraphicsResource_t res;
        cudaSurfaceObject_t surfObj;
        // Deal with async; TODO: kill this
        std::shared_ptr<CoreBuffer<float4>> rtBuffer;
        inline AuxRT(bool l, bool b) : 
            linked(l), bound(b), texture(nullptr), interested(false) {}
        inline AuxRT() : linked(false), bound(false), texture(nullptr), interested(false) {}
    };
    std::map<std::string, AuxRT> auxRTs;
    
public:
    // don't use comma in it
    inline void RegisterRT(const std::string& name) {
        assert(auxRTs.find(name) == auxRTs.end());
        auxRTs[name] = AuxRT(false, false);
    }

    // returns semicolon-separated list
    inline std::string ListRegisteredRTs() {
        std::string ret;
        for (auto &rt: auxRTs) {
            ret += rt.first + ";";
        }
        return ret;
    }

    inline bool SetInterest(const std::string &name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        it->second.interested = true;
    }

    inline bool ClearInterest(const std::string &name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        it->second.interested = false;
    }

    inline bool isInterested(const std::string &name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        return it->second.interested;
    }

    inline bool isSetupAndInterested(const std::string &name) {
        return isInterested(name) && isSetup(name);
    }

    inline CoreBuffer<float4>* getAssociatedBuffer(const std::string &name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return nullptr;
        }

        return it->second.rtBuffer.get();
    }

    std::map<std::string, AuxRT>::const_iterator begin() {
        return auxRTs.begin();
    }

    std::map<std::string, AuxRT>::const_iterator end() {
        return auxRTs.end();
    }

    // disable, but not removing
    // no-op on error
    inline bool DisableRT(const std::string& name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        if (it->second.bound) {
            CHK_CUDA(cudaGraphicsUnmapResources( 1, &it->second.res, /* STREAM */ 0 ));
            it->second.bound = false;
        }
        if (it->second.linked) {
            CHK_CUDA(cudaGraphicsUnregisterResource(it->second.res));
            it->second.linked = false;
        }
        return true;
    }

    // When main viewport changes
    inline void DisableAllRT() {
        for (auto &rt: auxRTs) {
            if (rt.second.bound) {
                CHK_CUDA(cudaGraphicsUnmapResources( 1, &rt.second.res, /* STREAM */ 0 ));
                rt.second.bound = false;
            }
            if (rt.second.linked) {
                CHK_CUDA(cudaGraphicsUnregisterResource(rt.second.res));
                rt.second.linked = false;
            }
        }
    }

    inline bool SetupTexture(const std::string& name, GLTexture* target) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            // fail sliently
            return false;
        }

        assert(!it->second.bound);
        DisableRT(name);
        it->second.texture = target;
        CHK_CUDA(cudaGraphicsGLRegisterImage(&it->second.res, target->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
        it->second.linked = true;

        const int newTargetPixels = target->height * target->width;
        if (it->second.rtBuffer.get() == nullptr ||
            it->second.rtBuffer->GetSize() < newTargetPixels) {
            it->second.rtBuffer = std::make_shared<CoreBuffer<float4>>(newTargetPixels, ON_DEVICE);
        }
        
        return true;
    }

    inline bool isSetup(const std::string& name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        if (it->second.linked == false) {
            return false;
        } else {
            return true;
        }
    }

    inline cudaSurfaceObject_t BindSurface(const std::string& name) {
        auto it = auxRTs.find(name);
        assert(it != auxRTs.end());
        assert(!it->second.bound);

        // Do sync
        cudaArray_t ar;
        CHK_CUDA(cudaGraphicsMapResources(1, &it->second.res, /* STREAM */ 0));
        it->second.bound = true;

        // (this call is intended to access one paticular mip level)
        // The value set in array may change every time that resource is mapped.
        CHK_CUDA(cudaGraphicsSubResourceGetMappedArray(&ar, it->second.res, 0, 0));

        cudaResourceDesc desc;
        desc.resType = cudaResourceType::cudaResourceTypeArray;
        desc.res.array.array = ar;

        CHK_CUDA(cudaCreateSurfaceObject(&it->second.surfObj, &desc));
        return it->second.surfObj;
    }
    
    inline void UnbindSurface(const std::string& name) {
        auto it = auxRTs.find(name);
        assert(it != auxRTs.end());
        assert(it->second.bound);

        CHK_CUDA(cudaDestroySurfaceObject(it->second.surfObj));
        CHK_CUDA(cudaGraphicsUnmapResources(1, &it->second.res, /* STREAM */ 0));
        it->second.bound = false;
    }
};
