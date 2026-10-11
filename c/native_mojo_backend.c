#include "native_mojo_backend.h"
#include <max/c/common.h>
#include <max/c/context.h>
#include <max/c/device.h>
#include <max/c/model.h>
#include <max/c/tensor.h>
#include <max/c/weights.h>
#include <max/c/safetensors.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct CMJEngine {
    M_RuntimeContext *context;
    M_Device *device,*host;
    M_AsyncCompiledModel *compiled;
    M_AsyncModel *model;
    M_Safetensors *weights;
    M_WeightsRegistry *registry;
    int accelerator_id;
    size_t limit;
};
static CMJStatus error(char *out,size_t cap,CMJStatus code,const char *text) {
    if(out&&cap)snprintf(out,cap,"%s",text?text:"native Mojo/MAX SDK failure");return code;
}
static int name_ok(const char *name) {
    if(!name||!*name)return 0;
    size_t n=0;while(name[n]&&n<=128){unsigned char c=(unsigned char)name[n++];if(c<32||c>126)return 0;}
    return n>0&&n<=128;
}
static M_Dtype sdk_dtype(CMJDType d) {
    switch(d) {
    case CMJ_F32:return M_FLOAT32;case CMJ_I8:return M_INT8;case CMJ_I16:return M_INT16;
    case CMJ_I4:return M_INT8;case CMJ_I64:return M_INT64;case CMJ_I32:return M_INT32;
    case CMJ_U8:return M_UINT8;case CMJ_F16:return M_FLOAT16;case CMJ_BF16:return M_BFLOAT16;
    default:return M_UNKNOWN;
    }
}
static CMJDType own_dtype(M_Dtype d) {
    for(unsigned n=CMJ_F32;n<=CMJ_BF16;n++)if(sdk_dtype((CMJDType)n)==d)return (CMJDType)n;
    return (CMJDType)0;
}
static int geometry(CMJDType dtype,size_t rank,const int64_t *shape,size_t *bytes,size_t *elements) {
    if(rank>CMJ_MAX_RANK||!shape)return 0;
    unsigned bits=dtype==CMJ_I4?4:(dtype==CMJ_I8||dtype==CMJ_U8)?8:
        (dtype==CMJ_I16||dtype==CMJ_F16||dtype==CMJ_BF16)?16:
        (dtype==CMJ_F32||dtype==CMJ_I32)?32:dtype==CMJ_I64?64:0;
    if(!bits)return 0;size_t count=1;
    for(size_t k=0;k<rank;k++){if(shape[k]<1||(uint64_t)shape[k]>SIZE_MAX/count)return 0;count*=(size_t)shape[k];}
    if(count>CMJ_MAX_BYTES*8/bits)return 0;
    *elements=count;*bytes=(count*bits+7)/8;return 1;
}
CMJStatus cmj_tensor_bytes(CMJDType d,size_t rank,const int64_t *shape,size_t *bytes) {
    size_t count=0,size=0;if(!bytes||!geometry(d,rank,shape,&size,&count))return CMJ_INVALID;
    *bytes=size;return CMJ_OK;
}
static int finite_data(CMJDType d,const void *data,size_t bytes) {
    const unsigned char *p=data;
    if(d==CMJ_F32)for(size_t k=0;k<bytes;k+=4){uint32_t u;memcpy(&u,p+k,4);if((u&0x7f800000u)==0x7f800000u)return 0;}
    if(d==CMJ_F16||d==CMJ_BF16)for(size_t k=0;k<bytes;k+=2){uint16_t u;memcpy(&u,p+k,2);unsigned mask=d==CMJ_F16?0x7c00u:0x7f80u;if((u&mask)==mask)return 0;}
    return 1;
}
const char *cmj_version(void){return M_version();}
int cmj_accelerator_count(void){return M_getAcceleratorCount();}
int cmj_host_available(void) {
    M_Status *s=M_newStatus();if(!s)return 0;
    M_Device *d=M_newDevice(M_HOST,0,s);int ok=d&&!M_isError(s)&&M_isHostDevice(d);
    M_freeDevice(d);M_freeStatus(s);return ok;
}
void cmj_destroy(CMJEngine *e) {
    if(!e)return;
    M_freeModel(e->model);M_freeWeightsRegistry(e->registry);M_freeSafetensors(e->weights);
    M_freeCompiledModel(e->compiled);M_freeRuntimeContext(e->context);
    M_freeDevice(e->device);M_freeDevice(e->host);free(e);
}
CMJStatus cmj_create(const CMJOptions *o,CMJEngine **out,char *message,size_t cap) {
    if(!o||!out||*out||!o->model_path||!*o->model_path||o->accelerator_id < -1||
       o->result_limit_bytes>CMJ_MAX_BYTES||o->weight_count>32||
       (o->weight_count&&!o->weight_paths))return error(message,cap,CMJ_INVALID,"invalid native Mojo options");
    for(size_t k=0;k<o->weight_count;k++)if(!o->weight_paths[k]||!*o->weight_paths[k])return error(message,cap,CMJ_INVALID,"invalid safetensor path");
    if(o->accelerator_id>=0&&o->accelerator_id>=cmj_accelerator_count())
        return error(message,cap,CMJ_DEVICE,"requested Mojo/MAX accelerator is unavailable; no CPU fallback");
    CMJEngine *e=calloc(1,sizeof *e);M_Status *s=M_newStatus();
    M_RuntimeConfig *r=NULL;M_CompileConfig *c=NULL;CMJStatus result=CMJ_SDK;
    if(!e||!s){result=CMJ_NOMEM;goto failed;}
    e->accelerator_id=o->accelerator_id;e->limit=o->result_limit_bytes?o->result_limit_bytes:CMJ_MAX_BYTES;
    e->device=M_newDevice(o->accelerator_id<0?M_HOST:M_ACCELERATOR,o->accelerator_id<0?0:o->accelerator_id,s);
    if(!e->device||M_isError(s))goto failed;
    e->host=M_newDevice(M_HOST,0,s);if(!e->host||M_isError(s))goto failed;
    r=M_newRuntimeConfig();if(!r){result=CMJ_NOMEM;goto failed;}
    M_runtimeConfigAddDevice(r,e->device);
    e->context=M_newRuntimeContext(r,s);M_freeRuntimeConfig(r);r=NULL;
    if(!e->context||M_isError(s))goto failed;
    c=M_newCompileConfig();if(!c){result=CMJ_NOMEM;goto failed;}
    M_setModelPath(c,o->model_path);
    e->compiled=M_compileModelSync(e->context,&c,s);
    if(!e->compiled||M_isError(s))goto failed;
    if(o->weight_count) {
        const char *paths[32];for(size_t k=0;k<o->weight_count;k++)paths[k]=o->weight_paths[k];
        e->weights=M_loadSafetensors(paths,o->weight_count,e->host,s);
        if(!e->weights||M_isError(s))goto failed;
        e->registry=M_newWeightsRegistryFromSafetensors(e->weights,s);
        if(!e->registry||M_isError(s))goto failed;
    }
    e->model=M_initModel(e->context,e->compiled,e->registry,s);
    if(!e->model||M_isError(s))goto failed;
    M_waitForModel(e->model,s);if(M_isError(s))goto failed;
    M_freeStatus(s);*out=e;if(message&&cap)*message=0;return CMJ_OK;
failed:
    error(message,cap,result,s&&M_isError(s)?M_getError(s):"native Mojo/MAX initialization failed");
    M_freeCompileConfig(c);M_freeRuntimeConfig(r);M_freeStatus(s);cmj_destroy(e);return result;
}
void cmj_output_free(CMJOutput *out,size_t count){if(out)for(size_t k=0;k<count;k++){free(out[k].data);memset(out+k,0,sizeof *out);}}
CMJStatus cmj_run(CMJEngine *e,const CMJInput *inputs,size_t count,
                  const char *const *names,size_t outputs,CMJOutput *out,char *message,size_t cap) {
    if(!e||!inputs||!count||count>CMJ_MAX_TENSORS||!names||!outputs||
       outputs>CMJ_MAX_TENSORS||!out)return error(message,cap,CMJ_INVALID,"invalid native tensor request");
    size_t total=0,expanded=0;
    for(size_t k=0;k<count;k++) {
        const CMJInput *in=inputs+k;size_t bytes=0,elements=0;
        if(!name_ok(in->name)||!geometry(in->dtype,in->rank,in->shape,&bytes,&elements)||
           !in->data||bytes!=in->bytes||bytes>CMJ_MAX_BYTES-total||!finite_data(in->dtype,in->data,bytes))
            return error(message,cap,CMJ_INVALID,"invalid shape, bytes, dtype or nonfinite input");
        total+=bytes;
        size_t runtime_bytes=in->dtype==CMJ_I4?elements:bytes;
        if(runtime_bytes>CMJ_MAX_BYTES-expanded)return error(message,cap,CMJ_LIMIT,"expanded input storage exceeds native budget");
        expanded+=runtime_bytes;
        for(size_t j=0;j<k;j++)if(!strcmp(in->name,inputs[j].name))return error(message,cap,CMJ_INVALID,"duplicate input tensor");
    }
    for(size_t k=0;k<outputs;k++) {
        if(!name_ok(names[k])||out[k].data)return error(message,cap,CMJ_INVALID,"invalid output name or occupied output");
        for(size_t j=0;j<k;j++)if(!strcmp(names[j],names[k]))return error(message,cap,CMJ_INVALID,"duplicate output name");
    }
    CMJOutput staged[CMJ_MAX_TENSORS]={0};M_Status *s=M_newStatus();
    void *owned_inputs[CMJ_MAX_TENSORS]={0};
    M_AsyncTensorMap *im=NULL,*om=NULL;M_AsyncTensor *tensor=NULL,*host_tensor=NULL;
    M_TensorSpec *spec=NULL;M_Device *source=NULL;CMJStatus result=CMJ_SDK;
    if(!s){result=CMJ_NOMEM;goto done;}
    im=M_newAsyncTensorMap(e->context);if(!im){result=CMJ_NOMEM;goto done;}
    for(size_t k=0;k<count;k++) {
        size_t bytes=0,elements=0;geometry(inputs[k].dtype,inputs[k].rank,inputs[k].shape,&bytes,&elements);
        size_t runtime_bytes=inputs[k].dtype==CMJ_I4?elements:bytes;
        owned_inputs[k]=malloc(runtime_bytes);if(!owned_inputs[k]){result=CMJ_NOMEM;goto done;}
        if(inputs[k].dtype==CMJ_I4)for(size_t j=0;j<elements;j++) {
            unsigned nibble=(((const unsigned char*)inputs[k].data)[j/2]>>(4*(j%2)))&15;
            ((int8_t*)owned_inputs[k])[j]=(int8_t)((int)(nibble^8)-8);
        }else memcpy(owned_inputs[k],inputs[k].data,bytes);
        spec=M_newTensorSpec(inputs[k].shape,(int64_t)inputs[k].rank,sdk_dtype(inputs[k].dtype),inputs[k].name,e->device);
        if(!spec){result=CMJ_NOMEM;goto done;}
        M_borrowTensorInto(im,owned_inputs[k],spec,s);
        M_freeTensorSpec(spec);spec=NULL;if(M_isError(s))goto done;
    }
    om=M_executeModelSync(e->context,e->model,im,s);
    if(!om||M_isError(s))goto done;total=0;
    M_synchronizeDevice(e->device,s);if(M_isError(s))goto done;
    for(size_t k=0;k<outputs;k++) {
        tensor=M_getTensorByNameFrom(om,names[k],s);if(!tensor||M_isError(s))goto done;
        source=M_getTensorDevice(tensor);if(!source)goto done;
        staged[k].source_accelerator=M_getDeviceType(source)==M_ACCELERATOR;
        staged[k].source_device_id=M_getDeviceId(source);
        if(staged[k].source_accelerator!=(e->accelerator_id>=0)||
           staged[k].source_device_id!=(e->accelerator_id<0?0:e->accelerator_id)) {
            error(message,cap,CMJ_DEVICE,"model output placement differs from explicitly requested device");result=CMJ_DEVICE;goto done;
        }
        M_freeDevice(source);source=NULL;
        host_tensor=M_copyTensorToDevice(tensor,e->host,s);
        if(!host_tensor||M_isError(s))goto done;
        M_synchronizeDevice(e->device,s);if(M_isError(s))goto done;
        M_synchronizeDevice(e->host,s);if(M_isError(s))goto done;
        spec=M_getTensorSpec(host_tensor);if(!spec)goto done;
        if(M_getDeviceTypeFromSpec(spec)!=M_HOST||M_getDeviceIdFromSpec(spec)!=0||
           M_getDtype(spec)!=M_getTensorType(host_tensor)){result=CMJ_DEVICE;error(message,cap,result,"SDK host-copy placement or dtype differs");goto done;}
        int64_t rank=M_getRank(spec);
        if(rank<0||rank>CMJ_MAX_RANK){result=CMJ_LIMIT;goto done;}
        staged[k].rank=(size_t)rank;staged[k].dtype=own_dtype(M_getTensorType(host_tensor));
        for(size_t j=0;j<staged[k].rank;j++)staged[k].shape[j]=M_getDimAt(spec,j);
        size_t elements=0;
        if(!geometry(staged[k].dtype,staged[k].rank,staged[k].shape,&staged[k].bytes,&elements)||
           elements!=M_getTensorNumElements(host_tensor)||staged[k].bytes>e->limit-total){result=CMJ_LIMIT;goto done;}
        total+=staged[k].bytes;
        const void *data=M_getTensorData(host_tensor);
        if(!data||!finite_data(staged[k].dtype,data,staged[k].bytes)){result=CMJ_INVALID;goto done;}
        staged[k].data=malloc(staged[k].bytes);if(!staged[k].data){result=CMJ_NOMEM;goto done;}
        memcpy(staged[k].data,data,staged[k].bytes);snprintf(staged[k].name,sizeof staged[k].name,"%s",names[k]);
        M_freeTensorSpec(spec);spec=NULL;M_freeTensor(host_tensor);host_tensor=NULL;M_freeTensor(tensor);tensor=NULL;
    }
    memcpy(out,staged,outputs*sizeof *out);memset(staged,0,sizeof staged);result=CMJ_OK;if(message&&cap)*message=0;
done:
    if(result!=CMJ_OK&&result!=CMJ_DEVICE)error(message,cap,result,s&&M_isError(s)?M_getError(s):"native Mojo/MAX tensor execution refused");
    M_freeDevice(source);M_freeTensorSpec(spec);M_freeTensor(host_tensor);M_freeTensor(tensor);
    if(im&&s)M_synchronizeDevice(e->device,s);
    M_freeAsyncTensorMap(om);M_freeAsyncTensorMap(im);M_freeStatus(s);
    for(size_t k=0;k<count;k++)free(owned_inputs[k]);
    cmj_output_free(staged,outputs);return result;
}
