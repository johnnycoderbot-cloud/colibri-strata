#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "native_graph_backend.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#define CNG_CALL __cdecl
#else
#include <unistd.h>
#include <dlfcn.h>
#define CNG_CALL
#endif

/* Minimal typed runtime bindings to the official OpenVINO C ABI and GenAI C
 * ABI2026.4.1 (source ab01d1771e4adaa0e1e08e9d630def170572506c). Only opaque
 * handles are declared locally, keeping fake-SDK tests/builds dependency-free.
 * ov_shape and ov_available_devices layouts below match installed SDK headers;
 * function argument counts/ownership are the published C API, not C++ mangling.
 */
typedef struct ov_core CNGOVCore;
typedef struct ov_model CNGOVModel;
typedef struct ov_compiled_model CNGOVCompiled;
typedef struct ov_infer_request CNGOVRequest;
typedef struct ov_tensor CNGOVTensor;
typedef struct ov_output_const_port CNGOVPort;
typedef struct ov_genai_llm_pipeline_opaque CNGLlm;
typedef struct ov_genai_generation_config_opaque CNGConfig;
typedef struct ov_genai_decoded_results_opaque CNGResult;
typedef struct ov_genai_whisper_pipeline_opaque CNGWhisper;
typedef struct ov_genai_whisper_generation_config_opaque CNGWhisperConfig;
typedef struct ov_genai_whisper_decoded_results_opaque CNGWhisperResult;
typedef struct { int64_t rank;int64_t *dims; } CNGShape;
typedef struct { char **devices;size_t size; } CNGDevices;
enum { CNG_OV_F32=4, CNG_OV_I64=10, CNG_OV_STRING=23 };
typedef struct {
    int (CNG_CALL *ov_core_create)(CNGOVCore**);
    void (CNG_CALL *ov_core_free)(CNGOVCore*);
    int (CNG_CALL *ov_core_get_available_devices)(const CNGOVCore*,CNGDevices*);
    void (CNG_CALL *ov_available_devices_free)(CNGDevices*);
    const char *(CNG_CALL *ov_get_last_err_msg)(void);
    void (CNG_CALL *ov_free)(const char*);
    int (CNG_CALL *ov_core_read_model)(const CNGOVCore*,const char*,const char*,CNGOVModel**);
    void (CNG_CALL *ov_model_free)(CNGOVModel*);
    int (CNG_CALL *ov_core_add_extension)(const CNGOVCore*,const char*);
    int (CNG_CALL *ov_model_inputs_size)(const CNGOVModel*,size_t*);
    int (CNG_CALL *ov_model_outputs_size)(const CNGOVModel*,size_t*);
    int (CNG_CALL *ov_model_const_input_by_name)(const CNGOVModel*,const char*,CNGOVPort**);
    int (CNG_CALL *ov_model_const_output_by_index)(const CNGOVModel*,size_t,CNGOVPort**);
    int (CNG_CALL *ov_const_port_get_shape)(const CNGOVPort*,CNGShape*);
    int (CNG_CALL *ov_port_get_element_type)(const CNGOVPort*,int*);
    void (CNG_CALL *ov_output_const_port_free)(CNGOVPort*);
    int (CNG_CALL *ov_shape_free)(CNGShape*);
    int (CNG_CALL *ov_core_compile_model)(const CNGOVCore*,const CNGOVModel*,const char*,size_t,CNGOVCompiled**,...);
    void (CNG_CALL *ov_compiled_model_free)(CNGOVCompiled*);
    int (CNG_CALL *ov_compiled_model_get_property)(const CNGOVCompiled*,const char*,char**);
    int (CNG_CALL *ov_compiled_model_create_infer_request)(const CNGOVCompiled*,CNGOVRequest**);
    void (CNG_CALL *ov_infer_request_free)(CNGOVRequest*);
    int (CNG_CALL *ov_tensor_create)(int,CNGShape,CNGOVTensor**);
    int (CNG_CALL *ov_tensor_create_from_string_array)(const char**,size_t,CNGShape,CNGOVTensor**);
    int (CNG_CALL *ov_tensor_data)(const CNGOVTensor*,void**);
    int (CNG_CALL *ov_tensor_get_shape)(const CNGOVTensor*,CNGShape*);
    int (CNG_CALL *ov_tensor_get_element_type)(const CNGOVTensor*,int*);
    int (CNG_CALL *ov_tensor_get_size)(const CNGOVTensor*,size_t*);
    void (CNG_CALL *ov_tensor_free)(CNGOVTensor*);
    int (CNG_CALL *ov_infer_request_set_tensor)(CNGOVRequest*,const char*,const CNGOVTensor*);
    int (CNG_CALL *ov_infer_request_set_input_tensor_by_index)(CNGOVRequest*,size_t,const CNGOVTensor*);
    int (CNG_CALL *ov_infer_request_get_tensor)(const CNGOVRequest*,const char*,CNGOVTensor**);
    int (CNG_CALL *ov_infer_request_get_output_tensor_by_index)(const CNGOVRequest*,size_t,CNGOVTensor**);
    int (CNG_CALL *ov_infer_request_infer)(CNGOVRequest*);
    int (CNG_CALL *ov_genai_llm_pipeline_create)(const char*,const char*,size_t,CNGLlm**,...);
    void (CNG_CALL *ov_genai_llm_pipeline_free)(CNGLlm*);
    int (CNG_CALL *ov_genai_llm_pipeline_generate)(CNGLlm*,const char*,const CNGConfig*,const void*,CNGResult**);
    int (CNG_CALL *ov_genai_llm_pipeline_get_generation_config)(const CNGLlm*,CNGConfig**);
    int (CNG_CALL *ov_genai_llm_pipeline_render_chat_template)(const CNGLlm*,const char*const*,const char*const*,size_t,char*,size_t*);
    void (CNG_CALL *ov_genai_generation_config_free)(CNGConfig*);
    int (CNG_CALL *ov_genai_generation_config_set_max_new_tokens)(CNGConfig*,size_t);
    int (CNG_CALL *ov_genai_generation_config_set_do_sample)(CNGConfig*,bool);
    int (CNG_CALL *ov_genai_generation_config_set_num_beams)(CNGConfig*,size_t);
    int (CNG_CALL *ov_genai_generation_config_set_num_return_sequences)(CNGConfig*,size_t);
    /* Reproducible extension to pinned official C wrapper; SDK already has
     * this setting internally. A missing export is an actionable component
     * error, never a reason to undercount implicitly wrapped prompts. */
    int (CNG_CALL *ov_genai_generation_config_set_apply_chat_template)(CNGConfig*,bool);
    int (CNG_CALL *ov_genai_decoded_results_get_string)(const CNGResult*,char*,size_t*);
    void (CNG_CALL *ov_genai_decoded_results_free)(CNGResult*);
    int (CNG_CALL *ov_genai_whisper_pipeline_create)(const char*,const char*,size_t,CNGWhisper**,...);
    void (CNG_CALL *ov_genai_whisper_pipeline_free)(CNGWhisper*);
    int (CNG_CALL *ov_genai_whisper_pipeline_generate)(CNGWhisper*,const float*,size_t,const CNGWhisperConfig*,CNGWhisperResult**);
    int (CNG_CALL *ov_genai_whisper_pipeline_get_generation_config)(const CNGWhisper*,CNGWhisperConfig**);
    int (CNG_CALL *ov_genai_whisper_generation_config_set_max_new_tokens)(CNGWhisperConfig*,size_t);
    int (CNG_CALL *ov_genai_whisper_generation_config_set_language)(CNGWhisperConfig*,const char*);
    int (CNG_CALL *ov_genai_whisper_generation_config_set_task)(CNGWhisperConfig*,const char*);
    void (CNG_CALL *ov_genai_whisper_generation_config_free)(CNGWhisperConfig*);
    int (CNG_CALL *ov_genai_whisper_decoded_results_get_texts_count)(const CNGWhisperResult*,size_t*);
    int (CNG_CALL *ov_genai_whisper_decoded_results_get_text_at)(const CNGWhisperResult*,size_t,char*,size_t*);
    void (CNG_CALL *ov_genai_whisper_decoded_results_free)(CNGWhisperResult*);
} CNGAPI;

struct CNGEngine {
    CNGAPI a;void *ov_dll,*genai_dll;
    CNGOVCore *core;CNGOVModel *model;CNGOVCompiled *compiled;CNGOVRequest *request;
    CNGOVTensor *lr,*bicubic;
    CNGLlm *llm;CNGWhisper *whisper;
    CNGInfo info;char *model_dir;
    size_t last_prompt_tokens,lr_count,bc_count;
};
#define CNG_PROMPT_MAX ((size_t)1 << 20)
#define CNG_OUTPUT_MAX ((size_t)8 << 20)
#define CNG_SR_MAX ((size_t)8 << 20) /* float elements per tile buffer */

static CNGStatus cng_error(CNGStatus s,char *error,size_t cap,const char *fmt,...) {
    if(error&&cap){va_list args;va_start(args,fmt);vsnprintf(error,cap,fmt,args);va_end(args);error[cap-1]=0;}return s;
}
static void cng_clear(char *error,size_t cap){if(error&&cap)error[0]=0;}
static size_t cng_length(const char *p,size_t bound) {size_t n=0;if(!p)return bound;while(n<bound&&p[n])n++;return n;}
static int cng_utf8(const char *p,size_t n) {
    size_t i=0;if(!p)return 0;
    while(i<n){unsigned c=(unsigned char)p[i++];if(!c)return 0;if(c<128)continue;
        unsigned cp,more;if(c>=0xc2&&c<=0xdf){cp=c&31;more=1;}else if(c>=0xe0&&c<=0xef){cp=c&15;more=2;}else if(c>=0xf0&&c<=0xf4){cp=c&7;more=3;}else return 0;
        if(more>n-i)return 0;for(unsigned j=0;j<more;j++){unsigned z=(unsigned char)p[i++];if((z&0xc0)!=0x80)return 0;cp=(cp<<6)|(z&63);}
        if((more==1&&cp<128)||(more==2&&cp<2048)||(more==3&&cp<65536)||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))return 0;
    }return 1;
}
static int cng_device_valid(const char *p) {
    size_t n=cng_length(p,64);if(!n||n==64||p[0]<'A'||p[0]>'Z')return 0;
    if(!strcmp(p,"AUTO")||!strcmp(p,"MULTI")||!strcmp(p,"HETERO")||!strcmp(p,"BATCH"))return 0;
    int suffix=0,digits=0;
    for(size_t i=1;i<n;i++){char c=p[i];if(c=='.'){if(suffix&&!digits)return 0;suffix=1;digits=0;}else if(suffix){if(c<'0'||c>'9')return 0;digits++;}else if(!((c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'))return 0;}
    return !suffix||digits;
}
static int cng_device_matches(const char *request,const char *actual) {
    if(!strcmp(request,actual))return 1;
    size_t n=strlen(request);return !strchr(request,'.')&&strncmp(request,actual,n)==0&&actual[n]=='.'&&cng_device_valid(actual);
}
static int cng_finite(const float *p,size_t n,int pixels) {
    if(!p)return 0;for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,p+i,4);if((u&0x7f800000u)==0x7f800000u)return 0;if(pixels&&(p[i]<0.f||p[i]>255.f))return 0;}return 1;
}
static char *cng_path(const char *dir,const char *name) {
    size_t n=cng_length(dir,4097),m=strlen(name);if(!n||n>4096)return NULL;
    char *p=malloc(n+m+2);if(p)snprintf(p,n+m+2,"%s/%s",dir,name);return p;
}
#ifndef CNG_LIBRARY_OPEN
static void *cng_library_open(const char *path,char *error,size_t cap) {
    if(!path||!path[0]||cng_length(path,4097)>4096){cng_error(CNG_NO_SDK,error,cap,"native C SDK library path is missing");return NULL;}
#ifdef _WIN32
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);wchar_t *w=n?malloc((size_t)n*sizeof(*w)):NULL;HMODULE h=NULL;
    if(w&&MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,w,n))h=LoadLibraryExW(w,NULL,LOAD_WITH_ALTERED_SEARCH_PATH);
    free(w);if(!h)cng_error(CNG_NO_SDK,error,cap,"cannot load native C SDK library (Windows error %lu)",(unsigned long)GetLastError());return (void*)h;
#else
    void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!h)cng_error(CNG_NO_SDK,error,cap,"cannot load native C SDK library: %s",dlerror());return h;
#endif
}
static void *cng_library_symbol(void *h,const char *name) {
#ifdef _WIN32
    FARPROC f=GetProcAddress((HMODULE)h,name);void *p=NULL;if(sizeof(p)==sizeof(f))memcpy(&p,&f,sizeof(p));return p;
#else
    return dlsym(h,name);
#endif
}
static void cng_library_close(void *h) {if(!h)return;
#ifdef _WIN32
    FreeLibrary((HMODULE)h);
#else
    dlclose(h);
#endif
}
#define CNG_LIBRARY_OPEN cng_library_open
#define CNG_LIBRARY_SYMBOL cng_library_symbol
#define CNG_LIBRARY_CLOSE cng_library_close
#endif
static int cng_bind(void *lib,const char *name,void *function,size_t size,char *error,size_t cap) {
    void *symbol=CNG_LIBRARY_SYMBOL(lib,name);
    if(!symbol||size!=sizeof(symbol)){cng_error(CNG_NO_SDK,error,cap,"native C SDK component lacks %s",name);return 0;}
    memcpy(function,&symbol,size);return 1;
}
#define CNG_BIND(e,lib,name) do{if(!cng_bind((lib),#name,&(e)->a.name,sizeof((e)->a.name),error,cap))return CNG_NO_SDK;}while(0)
static CNGStatus cng_load_core(CNGEngine *e,const char *path,char *error,size_t cap) {
    e->ov_dll=CNG_LIBRARY_OPEN(path,error,cap);if(!e->ov_dll)return CNG_NO_SDK;
    CNG_BIND(e,e->ov_dll,ov_core_create);CNG_BIND(e,e->ov_dll,ov_core_free);
    CNG_BIND(e,e->ov_dll,ov_core_get_available_devices);CNG_BIND(e,e->ov_dll,ov_available_devices_free);
    CNG_BIND(e,e->ov_dll,ov_get_last_err_msg);CNG_BIND(e,e->ov_dll,ov_free);return CNG_OK;
}
static CNGStatus cng_load_graph(CNGEngine *e,char *error,size_t cap) {
    CNG_BIND(e,e->ov_dll,ov_core_read_model);CNG_BIND(e,e->ov_dll,ov_model_free);
    CNG_BIND(e,e->ov_dll,ov_core_compile_model);CNG_BIND(e,e->ov_dll,ov_compiled_model_free);
    CNG_BIND(e,e->ov_dll,ov_compiled_model_create_infer_request);CNG_BIND(e,e->ov_dll,ov_infer_request_free);
    CNG_BIND(e,e->ov_dll,ov_tensor_free);CNG_BIND(e,e->ov_dll,ov_tensor_get_shape);
    CNG_BIND(e,e->ov_dll,ov_tensor_get_element_type);CNG_BIND(e,e->ov_dll,ov_tensor_get_size);
    CNG_BIND(e,e->ov_dll,ov_infer_request_infer);CNG_BIND(e,e->ov_dll,ov_shape_free);
    CNG_BIND(e,e->ov_dll,ov_model_inputs_size);CNG_BIND(e,e->ov_dll,ov_model_outputs_size);
    if(e->info.modality==CNG_TEXT) {
        CNG_BIND(e,e->ov_dll,ov_core_add_extension);CNG_BIND(e,e->ov_dll,ov_tensor_create_from_string_array);
        CNG_BIND(e,e->ov_dll,ov_infer_request_set_input_tensor_by_index);CNG_BIND(e,e->ov_dll,ov_infer_request_get_tensor);
    }else {
        CNG_BIND(e,e->ov_dll,ov_model_const_input_by_name);CNG_BIND(e,e->ov_dll,ov_model_const_output_by_index);
        CNG_BIND(e,e->ov_dll,ov_const_port_get_shape);CNG_BIND(e,e->ov_dll,ov_port_get_element_type);
        CNG_BIND(e,e->ov_dll,ov_output_const_port_free);CNG_BIND(e,e->ov_dll,ov_tensor_create);
        CNG_BIND(e,e->ov_dll,ov_tensor_data);CNG_BIND(e,e->ov_dll,ov_infer_request_set_tensor);
        CNG_BIND(e,e->ov_dll,ov_infer_request_get_output_tensor_by_index);CNG_BIND(e,e->ov_dll,ov_compiled_model_get_property);
    }return CNG_OK;
}
static CNGStatus cng_load_genai(CNGEngine *e,const char *path,char *error,size_t cap) {
    e->genai_dll=CNG_LIBRARY_OPEN(path,error,cap);if(!e->genai_dll)return CNG_NO_SDK;
    if(e->info.modality==CNG_TEXT) {
        CNG_BIND(e,e->genai_dll,ov_genai_llm_pipeline_create);CNG_BIND(e,e->genai_dll,ov_genai_llm_pipeline_free);
        CNG_BIND(e,e->genai_dll,ov_genai_llm_pipeline_generate);CNG_BIND(e,e->genai_dll,ov_genai_llm_pipeline_get_generation_config);
        CNG_BIND(e,e->genai_dll,ov_genai_llm_pipeline_render_chat_template);
        CNG_BIND(e,e->genai_dll,ov_genai_generation_config_free);CNG_BIND(e,e->genai_dll,ov_genai_generation_config_set_max_new_tokens);
        CNG_BIND(e,e->genai_dll,ov_genai_generation_config_set_do_sample);CNG_BIND(e,e->genai_dll,ov_genai_generation_config_set_num_beams);
        CNG_BIND(e,e->genai_dll,ov_genai_generation_config_set_num_return_sequences);CNG_BIND(e,e->genai_dll,ov_genai_decoded_results_get_string);
        CNG_BIND(e,e->genai_dll,ov_genai_generation_config_set_apply_chat_template);
        CNG_BIND(e,e->genai_dll,ov_genai_decoded_results_free);
    }else {
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_pipeline_create);CNG_BIND(e,e->genai_dll,ov_genai_whisper_pipeline_free);
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_pipeline_generate);CNG_BIND(e,e->genai_dll,ov_genai_whisper_pipeline_get_generation_config);
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_generation_config_set_max_new_tokens);
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_generation_config_set_language);CNG_BIND(e,e->genai_dll,ov_genai_whisper_generation_config_set_task);
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_generation_config_free);CNG_BIND(e,e->genai_dll,ov_genai_whisper_decoded_results_get_texts_count);
        CNG_BIND(e,e->genai_dll,ov_genai_whisper_decoded_results_get_text_at);CNG_BIND(e,e->genai_dll,ov_genai_whisper_decoded_results_free);
    }return CNG_OK;
}
static CNGStatus cng_sdk_error(CNGEngine *e,char *error,size_t cap,const char *operation) {
    const char *reason=e->a.ov_get_last_err_msg?e->a.ov_get_last_err_msg():NULL;
    return cng_error(CNG_SDK_ERROR,error,cap,"%s failed: %.768s",operation,reason?reason:"native SDK error");
}
void cng_free(void *p){free(p);}
void cng_destroy(CNGEngine *e) {
    if(!e)return;
    if(e->llm&&e->a.ov_genai_llm_pipeline_free)e->a.ov_genai_llm_pipeline_free(e->llm);
    if(e->whisper&&e->a.ov_genai_whisper_pipeline_free)e->a.ov_genai_whisper_pipeline_free(e->whisper);
    if(e->request&&e->a.ov_infer_request_free)e->a.ov_infer_request_free(e->request);
    if(e->lr&&e->a.ov_tensor_free)e->a.ov_tensor_free(e->lr);
    if(e->bicubic&&e->a.ov_tensor_free)e->a.ov_tensor_free(e->bicubic);
    if(e->compiled&&e->a.ov_compiled_model_free)e->a.ov_compiled_model_free(e->compiled);
    if(e->model&&e->a.ov_model_free)e->a.ov_model_free(e->model);
    if(e->core&&e->a.ov_core_free)e->a.ov_core_free(e->core);
    CNG_LIBRARY_CLOSE(e->genai_dll);CNG_LIBRARY_CLOSE(e->ov_dll);free(e->model_dir);free(e);
}
CNGStatus cng_info(const CNGEngine *e,CNGInfo *out){if(!e||!out)return CNG_INVALID;*out=e->info;return CNG_OK;}
CNGStatus cng_last_prompt_tokens(const CNGEngine *e,size_t *n){if(!e||!n||e->info.modality!=CNG_TEXT)return CNG_INVALID;*n=e->last_prompt_tokens;return CNG_OK;}
const char *cng_status_string(CNGStatus s) {
    static const char *const names[]={"ok","invalid input","limit exceeded","missing native C SDK component","device unavailable or substituted","native SDK error","nonfinite input/output","out of memory","unsupported native SDK feature"};
    return (unsigned)s<sizeof(names)/sizeof(names[0])?names[s]:"unknown error";
}
static CNGStatus cng_devices(CNGEngine *e,CNGDevices *d,char *error,size_t cap) {
    memset(d,0,sizeof(*d));if(e->a.ov_core_get_available_devices(e->core,d)!=0)return cng_sdk_error(e,error,cap,"native device enumeration");
    if(d->size>64||(d->size&&!d->devices))return cng_error(CNG_SDK_ERROR,error,cap,"native SDK returned an invalid device list");
    for(size_t i=0;i<d->size;i++)if(!cng_device_valid(d->devices[i]))return cng_error(CNG_SDK_ERROR,error,cap,"native SDK returned an invalid explicit device name");
    return CNG_OK;
}
CNGStatus cng_available_devices(const char *path,char **out,size_t *bytes,char *error,size_t cap) {
    cng_clear(error,cap);if(!out||*out||!bytes)return cng_error(CNG_INVALID,error,cap,"device output must initially be NULL");
    CNGEngine *e=calloc(1,sizeof(*e));if(!e)return cng_error(CNG_NOMEM,error,cap,"out of memory");
    CNGDevices d={0};CNGStatus s=cng_load_core(e,path,error,cap);char *text=NULL;
    if(s!=CNG_OK)goto done;if(e->a.ov_core_create(&e->core)!=0){s=cng_sdk_error(e,error,cap,"create native core");goto done;}
    s=cng_devices(e,&d,error,cap);if(s!=CNG_OK)goto done;
    size_t n=0;for(size_t i=0;i<d.size;i++)n+=strlen(d.devices[i])+1;text=malloc(n+1);
    if(!text){s=cng_error(CNG_NOMEM,error,cap,"out of memory");goto done;}size_t offset=0;
    for(size_t i=0;i<d.size;i++){size_t z=strlen(d.devices[i]);memcpy(text+offset,d.devices[i],z);offset+=z;text[offset++]='\n';}text[n]=0;
    *out=text;*bytes=n;text=NULL;
done:
    free(text);if(e->a.ov_available_devices_free)e->a.ov_available_devices_free(&d);cng_destroy(e);return s;
}
static CNGStatus cng_placement(CNGEngine *e,char *error,size_t cap) {
    e->info.execution_devices_verified=0;
    strcpy(e->info.selected_device,"unknown");
    char *property=NULL;int status=e->a.ov_compiled_model_get_property(e->compiled,"EXECUTION_DEVICES",&property);
    if(status||!property){if(property)e->a.ov_free(property);return cng_error(CNG_DEVICE,error,cap,"native graph execution devices cannot be verified");}
    size_t n=cng_length(property,128),begin=0,end=n;char device[64];int valid=n<128;
    while(begin<end&&(property[begin]==' '||property[begin]=='\t'))begin++;
    while(end>begin&&(property[end-1]==' '||property[end-1]=='\t'))end--;
    if(end>begin&&property[begin]=='['&&property[end-1]==']') {
        begin++;end--;
        while(begin<end&&(property[begin]==' '||property[begin]=='\t'))begin++;
        while(end>begin&&(property[end-1]==' '||property[end-1]=='\t'))end--;
    }
    if(end>begin&&(property[begin]=='\"'||property[begin]=='\'')&&property[end-1]==property[begin]){begin++;end--;}
    size_t used=end-begin;if(used>63)valid=0;
    if(valid)memcpy(device,property+begin,used);else used=0;
    device[used]=0;valid=valid&&cng_device_valid(device)&&cng_device_matches(e->info.requested_device,device);
    e->a.ov_free(property);if(!valid)return cng_error(CNG_DEVICE,error,cap,"native graph substituted or combined an unexpected execution device");
    snprintf(e->info.selected_device,sizeof(e->info.selected_device),"%s",device);e->info.execution_devices_verified=1;return CNG_OK;
}
static int cng_port_shape(CNGEngine *e,int input,const char *name,size_t *width,size_t *height) {
    CNGOVPort *port=NULL;CNGShape shape={0};int type=-1;int okay=0;
    if((input?e->a.ov_model_const_input_by_name(e->model,name,&port):e->a.ov_model_const_output_by_index(e->model,0,&port))!=0||!port)goto done;
    if(e->a.ov_port_get_element_type(port,&type)||type!=CNG_OV_F32||e->a.ov_const_port_get_shape(port,&shape))goto done;
    if(shape.rank!=4||!shape.dims||shape.dims[0]!=1||shape.dims[1]!=3||shape.dims[2]<1||shape.dims[3]<1||shape.dims[2]>16384||shape.dims[3]>16384)goto done;
    size_t h=(size_t)shape.dims[2],w=(size_t)shape.dims[3];if(w>CNG_SR_MAX/(3*h))goto done;*width=w;*height=h;okay=1;
done:
    if(shape.dims)e->a.ov_shape_free(&shape);if(port)e->a.ov_output_const_port_free(port);return okay;
}
static CNGStatus cng_graph_create(CNGEngine *e,const CNGOptions *o,char *error,size_t cap) {
    CNGStatus s=cng_load_graph(e,error,cap);if(s!=CNG_OK)return s;
    char *xml=NULL,*extension=NULL;const char *device=o->device;
    if(o->modality==CNG_TEXT) {
        xml=cng_path(o->model_dir,"openvino_tokenizer.xml");device="CPU";
        if(o->tokenizers_library){size_t n=cng_length(o->tokenizers_library,4097);if(n<=4096){extension=malloc(n+1);if(extension)memcpy(extension,o->tokenizers_library,n+1);}}
        else {
            const char *p=o->genai_c_library,*slash=p?strrchr(p,'/'):NULL,*back=p?strrchr(p,'\\'):NULL;
            if(back&&(!slash||back>slash))slash=back;
            if(slash){size_t n=(size_t)(slash-p);char *dir=malloc(n+1);if(dir){memcpy(dir,p,n);dir[n]=0;
#ifdef _WIN32
                extension=cng_path(dir,"openvino_tokenizers.dll");
#else
                extension=cng_path(dir,"libopenvino_tokenizers.so");
#endif
                free(dir);}}
        }
        if(!extension||e->a.ov_core_add_extension(e->core,extension)){free(extension);free(xml);return cng_error(CNG_NO_SDK,error,cap,"native tokenizer extension is missing or incompatible");}free(extension);
    }else if(o->sr_xml){size_t n=cng_length(o->sr_xml,4097);if(n<=4096){xml=malloc(n+1);if(xml)memcpy(xml,o->sr_xml,n+1);}}
    if(!xml)return cng_error(CNG_INVALID,error,cap,"native graph model path is missing");
    int code=e->a.ov_core_read_model(e->core,xml,NULL,&e->model);free(xml);if(code||!e->model)return cng_sdk_error(e,error,cap,"read native graph");
    size_t inputs=0,outputs=0;
    if(e->a.ov_model_inputs_size(e->model,&inputs)||e->a.ov_model_outputs_size(e->model,&outputs))return cng_sdk_error(e,error,cap,"inspect native graph");
    if(o->modality==CNG_CAMERA) {
        if(inputs!=2||outputs!=1||!cng_port_shape(e,1,"0",&e->info.lr_width,&e->info.lr_height)||
           !cng_port_shape(e,1,"1",&e->info.bicubic_width,&e->info.bicubic_height)||
           !cng_port_shape(e,0,NULL,&e->info.output_width,&e->info.output_height))return cng_error(CNG_INVALID,error,cap,"trained SR graph requires static F32 planarBGR inputs0/1 and output");
        size_t w=e->info.output_width,h=e->info.output_height;
        if(w!=e->info.bicubic_width||h!=e->info.bicubic_height||w%e->info.lr_width||h%e->info.lr_height||
           w/e->info.lr_width!=h/e->info.lr_height||w/e->info.lr_width<2||w/e->info.lr_width>8)return cng_error(CNG_INVALID,error,cap,"trained SR graph has inconsistent tile/scale geometry");
        e->info.scale=(unsigned)(w/e->info.lr_width);e->lr_count=3*e->info.lr_width*e->info.lr_height;e->bc_count=3*w*h;
    }else if(inputs!=1)return cng_error(CNG_UNSUPPORTED,error,cap,"native tokenizer graph must have one STRING input");
    if(o->modality==CNG_CAMERA&&(!strcmp(device,"CPU")||!strncmp(device,"GPU",3)))code=e->a.ov_core_compile_model(e->core,e->model,device,2,&e->compiled,"INFERENCE_PRECISION_HINT",o->inference_f16?"f16":"f32");
    else code=e->a.ov_core_compile_model(e->core,e->model,device,0,&e->compiled);
    if(code||!e->compiled)return cng_sdk_error(e,error,cap,"compile native graph on explicit device");
    if(e->a.ov_compiled_model_create_infer_request(e->compiled,&e->request)||!e->request)return cng_sdk_error(e,error,cap,"create native inference request");
    if(o->modality==CNG_CAMERA) {
        s=cng_placement(e,error,cap);if(s!=CNG_OK)return s;char *precision=NULL;
        if(!e->a.ov_compiled_model_get_property(e->compiled,"INFERENCE_PRECISION_HINT",&precision)&&precision&&cng_length(precision,64)<64)snprintf(e->info.runtime_precision,sizeof(e->info.runtime_precision),"%s",precision);
        if(precision)e->a.ov_free(precision);
        int64_t d0[4]={1,3,(int64_t)e->info.lr_height,(int64_t)e->info.lr_width};CNGShape sh0={4,d0};
        int64_t d1[4]={1,3,(int64_t)e->info.bicubic_height,(int64_t)e->info.bicubic_width};CNGShape sh1={4,d1};
        if(e->a.ov_tensor_create(CNG_OV_F32,sh0,&e->lr)||e->a.ov_tensor_create(CNG_OV_F32,sh1,&e->bicubic)||
           e->a.ov_infer_request_set_tensor(e->request,"0",e->lr)||e->a.ov_infer_request_set_tensor(e->request,"1",e->bicubic))return cng_sdk_error(e,error,cap,"allocate native SR input tensors");
    }return CNG_OK;
}
CNGStatus cng_create(const CNGOptions *o,CNGEngine **out,char *error,size_t cap) {
    cng_clear(error,cap);if(!o||!out||*out||!cng_device_valid(o->device)||!o->model_dir||!o->model_dir[0]||
       o->modality<CNG_TEXT||o->modality>CNG_CAMERA||(o->inference_f16!=0&&o->inference_f16!=1))return cng_error(CNG_INVALID,error,cap,"invalid native engine options or explicit device");
    unsigned limit=o->modality==CNG_AUDIO?448:256,tokens=o->max_tokens?o->max_tokens:limit;
    unsigned prompt=o->max_prompt_tokens?o->max_prompt_tokens:2048,context=o->max_context_tokens?o->max_context_tokens:4096;
    if(tokens>limit||prompt>4096||context<2||context>4096||prompt>context)return cng_error(CNG_LIMIT,error,cap,"native token limits are out of range");
    CNGEngine *e=calloc(1,sizeof(*e));if(!e)return cng_error(CNG_NOMEM,error,cap,"out of memory");
    e->info.modality=o->modality;e->info.max_tokens=tokens;e->info.max_prompt_tokens=prompt;e->info.max_context_tokens=context;
    snprintf(e->info.requested_device,sizeof(e->info.requested_device),"%s",o->device);snprintf(e->info.selected_device,sizeof(e->info.selected_device),"%s",o->device);strcpy(e->info.runtime_precision,"unknown");
    size_t dirn=cng_length(o->model_dir,4097);if(dirn>4096){cng_destroy(e);return cng_error(CNG_LIMIT,error,cap,"model path exceeds its bound");}
    e->model_dir=malloc(dirn+1);if(!e->model_dir){cng_destroy(e);return cng_error(CNG_NOMEM,error,cap,"out of memory");}memcpy(e->model_dir,o->model_dir,dirn+1);
    CNGStatus s=cng_load_core(e,o->ov_c_library,error,cap);CNGDevices d={0};if(s!=CNG_OK)goto done;
    if(e->a.ov_core_create(&e->core)||!e->core){s=cng_sdk_error(e,error,cap,"create native core");goto done;}
    s=cng_devices(e,&d,error,cap);if(s!=CNG_OK)goto done;int found=0;
    for(size_t i=0;i<d.size;i++)if(cng_device_matches(o->device,d.devices[i]))found=1;
    if(!found){s=cng_error(CNG_DEVICE,error,cap,"requested native device %s is unavailable",o->device);goto done;}
    if(o->modality!=CNG_CAMERA) {
        s=cng_load_genai(e,o->genai_c_library,error,cap);if(s!=CNG_OK)goto done;
        int code=o->modality==CNG_TEXT?e->a.ov_genai_llm_pipeline_create(o->model_dir,o->device,0,&e->llm):e->a.ov_genai_whisper_pipeline_create(o->model_dir,o->device,0,&e->whisper);
        if(code||(o->modality==CNG_TEXT?!e->llm:!e->whisper)){s=cng_sdk_error(e,error,cap,"create native GenAI pipeline on explicit device");goto done;}
    }
    if(o->modality!=CNG_AUDIO){s=cng_graph_create(e,o,error,cap);if(s!=CNG_OK)goto done;}
    *out=e;e=NULL;s=CNG_OK;
done:
    if(e&&e->a.ov_available_devices_free)e->a.ov_available_devices_free(&d);
    else if(!e){ /* Use the still-live published engine to release SDK list. */
        (*out)->a.ov_available_devices_free(&d);
    }cng_destroy(e);return s;
}
static CNGStatus cng_prompt_count(CNGEngine *e,const char *prompt,size_t *count,char *error,size_t cap) {
    int64_t dim=1;CNGShape shape={1,&dim};const char *strings[1]={prompt};CNGOVTensor *input=NULL,*ids=NULL;CNGShape got={0};CNGStatus status=CNG_SDK_ERROR;
    if(e->a.ov_tensor_create_from_string_array(strings,1,shape,&input)||!input||
       e->a.ov_infer_request_set_input_tensor_by_index(e->request,0,input)||e->a.ov_infer_request_infer(e->request)||
       e->a.ov_infer_request_get_tensor(e->request,"input_ids",&ids)||!ids||e->a.ov_tensor_get_shape(ids,&got))goto done;
    int type=-1;size_t n=0;
    if(got.rank!=2||!got.dims||got.dims[0]!=1||got.dims[1]<1||got.dims[1]>(int64_t)CNG_PROMPT_MAX||
       e->a.ov_tensor_get_element_type(ids,&type)||type!=CNG_OV_I64||e->a.ov_tensor_get_size(ids,&n)||n!=(size_t)got.dims[1])goto done;
    *count=n;status=CNG_OK;
done:
    if(got.dims)e->a.ov_shape_free(&got);if(ids)e->a.ov_tensor_free(ids);if(input)e->a.ov_tensor_free(input);
    return status==CNG_OK?status:cng_sdk_error(e,error,cap,"native tokenizer count");
}
static CNGStatus cng_result_validate(char *text,size_t allocated,size_t actual,char **out,size_t *bytes,char *error,size_t cap) {
    if(!actual||actual>allocated||text[actual-1]||cng_length(text,actual)!=actual-1||!cng_utf8(text,actual-1))return cng_error(CNG_SDK_ERROR,error,cap,"native SDK returned malformed UTF8 output");
    *out=text;*bytes=actual-1;return CNG_OK;
}
CNGStatus cng_render_chat(CNGEngine *e,const char *const *roles,const char *const *contents,size_t count,char **out,size_t *bytes,char *error,size_t cap) {
    cng_clear(error,cap);
    if(!e||e->info.modality!=CNG_TEXT||!roles||!contents||!count||count>64||!out||*out||!bytes)return cng_error(CNG_INVALID,error,cap,"invalid native chat request");
    size_t total=0;
    for(size_t i=0;i<count;i++) {
        size_t r=cng_length(roles[i],16),n=cng_length(contents[i],CNG_PROMPT_MAX+1);
        if(r>=16||!roles[i]||(strcmp(roles[i],"system")&&strcmp(roles[i],"user")&&strcmp(roles[i],"assistant"))||!contents[i]||!cng_utf8(contents[i],n))return cng_error(CNG_INVALID,error,cap,"chat messages require valid UTF8 roles/content");
        if(n>CNG_PROMPT_MAX||total>CNG_PROMPT_MAX-n)return cng_error(CNG_LIMIT,error,cap,"chat content exceeds native byte limit");total+=n;
    }
    size_t needed=0;
    int code=e->a.ov_genai_llm_pipeline_render_chat_template(e->llm,roles,contents,count,NULL,&needed);
    if(code==-6)return cng_error(CNG_LIMIT,error,cap,"rendered model chat template exceeds native byte limit");
    if(code)return cng_sdk_error(e,error,cap,"native model chat template");
    if(!needed||needed>CNG_PROMPT_MAX+1)return cng_error(CNG_LIMIT,error,cap,"rendered model chat template exceeds native byte limit");
    char *text=malloc(needed);if(!text)return cng_error(CNG_NOMEM,error,cap,"out of memory");memset(text,0xa5,needed);size_t actual=needed;
    if(e->a.ov_genai_llm_pipeline_render_chat_template(e->llm,roles,contents,count,text,&actual)){free(text);return cng_sdk_error(e,error,cap,"render native model chat template");}
    CNGStatus status=cng_result_validate(text,needed,actual,out,bytes,error,cap);if(status!=CNG_OK)free(text);return status;
}
CNGStatus cng_text(CNGEngine *e,const char *prompt,size_t n,unsigned tokens,char **out,size_t *bytes,char *error,size_t cap) {
    cng_clear(error,cap);if(!e||e->info.modality!=CNG_TEXT||!out||*out||!bytes||!n||n>CNG_PROMPT_MAX||!cng_utf8(prompt,n))return cng_error(CNG_INVALID,error,cap,"invalid bounded UTF8 text request");
    if(!tokens)tokens=e->info.max_tokens;if(tokens>e->info.max_tokens)return cng_error(CNG_LIMIT,error,cap,"generation exceeds native token limit");
    char *p=malloc(n+1);if(!p)return cng_error(CNG_NOMEM,error,cap,"out of memory");memcpy(p,prompt,n);p[n]=0;
    size_t count=0;CNGStatus s=cng_prompt_count(e,p,&count,error,cap);if(s!=CNG_OK){free(p);return s;}
    if(count>e->info.max_prompt_tokens||count+tokens>e->info.max_context_tokens){free(p);return cng_error(CNG_LIMIT,error,cap,"prompt or prompt+generation exceeds native context limit");}
    CNGConfig *config=NULL;CNGResult *result=NULL;char *text=NULL;
    int code=e->a.ov_genai_llm_pipeline_get_generation_config(e->llm,&config);
    if(code||!config||e->a.ov_genai_generation_config_set_max_new_tokens(config,tokens)||
       e->a.ov_genai_generation_config_set_do_sample(config,false)||e->a.ov_genai_generation_config_set_num_beams(config,1)||
       e->a.ov_genai_generation_config_set_num_return_sequences(config,1)||
       e->a.ov_genai_generation_config_set_apply_chat_template(config,false)||
       e->a.ov_genai_llm_pipeline_generate(e->llm,p,config,NULL,&result)||!result){s=cng_sdk_error(e,error,cap,"native stateless text generation");goto done;}
    size_t needed=0;if(e->a.ov_genai_decoded_results_get_string(result,NULL,&needed)||!needed||needed>CNG_OUTPUT_MAX){s=cng_error(CNG_LIMIT,error,cap,"native text result exceeds its bound");goto done;}
    text=malloc(needed);if(!text){s=cng_error(CNG_NOMEM,error,cap,"out of memory");goto done;}memset(text,0xa5,needed);size_t actual=needed;
    if(e->a.ov_genai_decoded_results_get_string(result,text,&actual)){s=cng_sdk_error(e,error,cap,"read native text result");goto done;}
    s=cng_result_validate(text,needed,actual,out,bytes,error,cap);if(s==CNG_OK){text=NULL;e->last_prompt_tokens=count;}
done:
    free(text);free(p);if(result)e->a.ov_genai_decoded_results_free(result);if(config)e->a.ov_genai_generation_config_free(config);return s;
}
CNGStatus cng_whisper(CNGEngine *e,const float *samples,size_t n,const char *task,const char *language,char **out,size_t *bytes,char *error,size_t cap) {
    cng_clear(error,cap);if(!e||e->info.modality!=CNG_AUDIO||!out||*out||!bytes||!samples||!n||n>=480000||
       !task||(strcmp(task,"transcribe")&&strcmp(task,"translate"))||(language&&strcmp(language,"en")))return cng_error(CNG_INVALID,error,cap,"invalid mono16k/task/language speech request");
    if(!cng_finite(samples,n,0))return cng_error(CNG_NONFINITE,error,cap,"speech input contains nonfinite samples");
    for(size_t i=0;i<n;i++)if(samples[i]<-1.f||samples[i]>1.f)return cng_error(CNG_INVALID,error,cap,"speech samples must be normalized to[-1,1]");
    CNGWhisperConfig *config=NULL;CNGWhisperResult *result=NULL;char *text=NULL;CNGStatus s=CNG_OK;
    int code=e->a.ov_genai_whisper_pipeline_get_generation_config(e->whisper,&config);
    if(code||!config||e->a.ov_genai_whisper_generation_config_set_max_new_tokens(config,e->info.max_tokens)||
       e->a.ov_genai_whisper_generation_config_set_language(config,language?"<|en|>":NULL)||
       e->a.ov_genai_whisper_generation_config_set_task(config,task)||e->a.ov_genai_whisper_pipeline_generate(e->whisper,samples,n,config,&result)||!result){s=cng_sdk_error(e,error,cap,"native Whisper generation");goto done;}
    size_t count=0,needed=0;if(e->a.ov_genai_whisper_decoded_results_get_texts_count(result,&count)||count!=1||
       e->a.ov_genai_whisper_decoded_results_get_text_at(result,0,NULL,&needed)||!needed||needed>CNG_OUTPUT_MAX){s=cng_error(CNG_SDK_ERROR,error,cap,"native Whisper result shape/size is invalid");goto done;}
    text=malloc(needed);if(!text){s=cng_error(CNG_NOMEM,error,cap,"out of memory");goto done;}memset(text,0xa5,needed);size_t actual=needed;
    if(e->a.ov_genai_whisper_decoded_results_get_text_at(result,0,text,&actual)){s=cng_sdk_error(e,error,cap,"read native Whisper result");goto done;}
    s=cng_result_validate(text,needed,actual,out,bytes,error,cap);if(s==CNG_OK)text=NULL;
done:
    free(text);if(result)e->a.ov_genai_whisper_decoded_results_free(result);if(config)e->a.ov_genai_whisper_generation_config_free(config);return s;
}
CNGStatus cng_sr(CNGEngine *e,const float *lr,size_t ln,const float *bc,size_t bn,float **out,size_t *count,char *error,size_t cap) {
    cng_clear(error,cap);if(!e||e->info.modality!=CNG_CAMERA||!out||*out||!count||ln!=e->lr_count||bn!=e->bc_count)return cng_error(CNG_INVALID,error,cap,"trained SR requires exact planarBGR tile sizes");
    if(!cng_finite(lr,ln,1)||!cng_finite(bc,bn,1))return cng_error(CNG_NONFINITE,error,cap,"SR inputs must be finite BGR values0..255");
    void *l=NULL,*b=NULL;CNGOVTensor *result=NULL;CNGShape shape={0};float *copy=NULL;CNGStatus s=CNG_OK;
    if(e->a.ov_tensor_data(e->lr,&l)||!l||e->a.ov_tensor_data(e->bicubic,&b)||!b){s=cng_sdk_error(e,error,cap,"access native SR input tensors");goto done;}
    memcpy(l,lr,ln*4);memcpy(b,bc,bn*4);
    if(e->a.ov_infer_request_infer(e->request)||e->a.ov_infer_request_get_output_tensor_by_index(e->request,0,&result)||!result||e->a.ov_tensor_get_shape(result,&shape)){s=cng_sdk_error(e,error,cap,"native trained SR inference");goto done;}
    int type=-1;size_t n=0;void *data=NULL;
    if(shape.rank!=4||!shape.dims||shape.dims[0]!=1||shape.dims[1]!=3||shape.dims[2]!=(int64_t)e->info.output_height||shape.dims[3]!=(int64_t)e->info.output_width||
       e->a.ov_tensor_get_element_type(result,&type)||type!=CNG_OV_F32||e->a.ov_tensor_get_size(result,&n)||n!=bn||e->a.ov_tensor_data(result,&data)||!data){s=cng_error(CNG_SDK_ERROR,error,cap,"native SR output shape/type/size is invalid");goto done;}
    if(!cng_finite(data,n,0)){s=cng_error(CNG_NONFINITE,error,cap,"native trained SR output is nonfinite");goto done;}
    s=cng_placement(e,error,cap);if(s!=CNG_OK)goto done;
    copy=malloc(n*4);if(!copy){s=cng_error(CNG_NOMEM,error,cap,"out of memory");goto done;}memcpy(copy,data,n*4);*out=copy;*count=n;copy=NULL;
done:
    free(copy);if(shape.dims)e->a.ov_shape_free(&shape);if(result)e->a.ov_tensor_free(result);return s;
}
