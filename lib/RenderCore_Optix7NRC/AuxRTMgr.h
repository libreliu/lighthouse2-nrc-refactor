#include "core_settings.h"
#include <vector>
#include <map>
#include <string>

// This class uses the SOTA Surface Object API
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
class AuxRTMgr {
private:
    struct AuxRT {
        bool linked;
        bool bound;
        GLTexture *texture;
        cudaGraphicsResource_t res;
        cudaSurfaceObject_t surfObj;
        inline AuxRT(bool l, bool b) : linked(l), bound(b) {}
    };
    std::map<std::string, AuxRT> auxRTs;
    
public:
    inline void RegisterRT(const std::string& name) {
        assert(auxRTs.find(name) == auxRTs.end());
        auxRTs[name] = AuxRT(false, false);
    }

    // disable, but not removing
    // no-op on error
    inline void DisableRT(const std::string& name) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return;
        }

        if (it->second.bound) {
            CHK_CUDA(cudaGraphicsUnmapResources( 1, &it->second.res, /* STREAM */ 0 ));
            it->second.bound = false;
        }
        if (it->second.linked) {
            CHK_CUDA(cudaGraphicsUnregisterResource(it->second.res));
            it->second.linked = false;
        }
    }

    inline bool SetupTexture(const std::string& name, GLTexture* target) {
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return false;
        }

        DisableRT(name);
        it->second.texture = target;
        CHK_CUDA(cudaGraphicsGLRegisterImage(&it->second.res, target->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));


    }

    inline void BindSurface(const std::string& name) {
        struct cudaResourceDesc resDesc;
        auto it = auxRTs.find(name);
        if (it == auxRTs.end()) {
            return;
        }

        // Do sync
        cudaArray_t ar;
        CHK_CUDA(cudaGraphicsMapResources(1, &it->second.res, /* STREAM */ 0));

        // (this call is intended to access one paticular mip level)
        // The value set in array may change every time that resource is mapped.
        CHK_CUDA(cudaGraphicsSubResourceGetMappedArray(&ar, it->second.res, 0, 0));
    }
    
    inline void UnbindSurface(const std::string& name) {

    }

};
