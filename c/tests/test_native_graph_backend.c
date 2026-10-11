/* Isolated native C SDK fakes: ABI dispatch/ownership/limits/placement gates.
 * Fake values prove binding behavior only; trained-model replay is separate. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include <stddef.h>
static void *fake_open(const char *,char *,size_t);
static void *fake_symbol(void *,const char *);
static void fake_close(void *);
#define CNG_LIBRARY_OPEN fake_open
#define CNG_LIBRARY_SYMBOL fake_symbol
#define CNG_LIBRARY_CLOSE fake_close
#include "../native_graph_backend.c"
#include <assert.h>

struct ov_core { int live; };
struct ov_model { int tokenizer; };
struct ov_compiled_model { int tokenizer;char device[64]; };
struct ov_output_const_port { int index; };
struct ov_tensor { int refs,type;int64_t dims[4],rank;size_t count;void *data; };
struct ov_infer_request { int tokenizer;CNGOVTensor *in[2]; };
struct ov_genai_llm_pipeline_opaque { char device[64]; };
struct ov_genai_whisper_pipeline_opaque { char device[64]; };
struct ov_genai_generation_config_opaque { size_t tokens,beams,returns;bool sample,apply;int model_eos; };
struct ov_genai_whisper_generation_config_opaque { unsigned tokens;char task[16],language[16];int model_eos; };
struct ov_genai_decoded_results_opaque { int placeholder; };
struct ov_genai_whisper_decoded_results_opaque { int placeholder; };
static int handles,libraries,text_calls,audio_calls,sr_calls,token_calls;
static int fault,token_size=7;static const char *missing_symbol;
enum { F_PLACEMENT=1,F_PRECISION,F_OUTPUT_NAN,F_OUTPUT_SHAPE,F_OUTPUT_TYPE,F_BAD_TEXT,F_TEXT_SIZE,F_NO_NUL,F_INFER,F_DEVICE_LIST,F_META,F_CONFIG };
static void *alloc_handle(size_t n){void *p=calloc(1,n);assert(p);handles++;return p;}
static void free_handle(void *p){if(p){handles--;free(p);}}
static char *text_copy(const char *p){size_t n=strlen(p)+1;char *r=malloc(n);assert(r);memcpy(r,p,n);return r;}
static int ov_core_create(CNGOVCore **p){*p=alloc_handle(sizeof(**p));return 0;}
static void ov_core_free(CNGOVCore *p){free_handle(p);}
static const char *ov_get_last_err_msg(void){return "injected native SDK failure";}
static void ov_free(const char *p){free((void*)p);}
static int ov_core_get_available_devices(const CNGOVCore *p,CNGDevices *d){assert(p);d->size=3;d->devices=calloc(3,sizeof(char*));assert(d->devices);d->devices[0]=text_copy("CPU");d->devices[1]=text_copy("GPU.0");d->devices[2]=text_copy(fault==F_DEVICE_LIST?"AUTO":"GPU.1");return 0;}
static void ov_available_devices_free(CNGDevices *d){for(size_t i=0;i<d->size;i++)free(d->devices[i]);free(d->devices);memset(d,0,sizeof(*d));}
static int ov_core_read_model(const CNGOVCore *p,const char *name,const char *bin,CNGOVModel **out){assert(p&&name&&!bin);*out=alloc_handle(sizeof(**out));(*out)->tokenizer=strstr(name,"tokenizer")!=NULL;return 0;}
static void ov_model_free(CNGOVModel *m){free_handle(m);}
static int ov_core_add_extension(const CNGOVCore *p,const char *name){assert(p&&name&&strstr(name,"tokenizers"));return 0;}
static int ov_model_inputs_size(const CNGOVModel *m,size_t *n){*n=m->tokenizer?1:2;return 0;}
static int ov_model_outputs_size(const CNGOVModel *m,size_t *n){*n=m->tokenizer?2:1;return 0;}
static int ov_model_const_input_by_name(const CNGOVModel *m,const char *name,CNGOVPort **out){assert(!m->tokenizer);if(strcmp(name,"0")&&strcmp(name,"1"))return -1;*out=alloc_handle(sizeof(**out));(*out)->index=name[0]=='0'?0:1;return 0;}
static int ov_model_const_output_by_index(const CNGOVModel *m,size_t i,CNGOVPort **out){assert(!m->tokenizer&&i==0);*out=alloc_handle(sizeof(**out));(*out)->index=2;return 0;}
static int ov_const_port_get_shape(const CNGOVPort *p,CNGShape *s){s->rank=4;s->dims=malloc(32);assert(s->dims);s->dims[0]=1;s->dims[1]=3;s->dims[2]=p->index?8:2;s->dims[3]=p->index?12:3;if(fault==F_META)s->dims[3]=16385;return 0;}
static int ov_port_get_element_type(const CNGOVPort *p,int *t){assert(p);*t=CNG_OV_F32;return 0;}
static void ov_output_const_port_free(CNGOVPort *p){free_handle(p);}
static int ov_shape_free(CNGShape *s){free(s->dims);s->dims=NULL;s->rank=0;return 0;}
static int ov_core_compile_model(const CNGOVCore *core,const CNGOVModel *m,const char *device,size_t nargs,CNGOVCompiled **out,...){
    assert(core);*out=alloc_handle(sizeof(**out));(*out)->tokenizer=m->tokenizer;snprintf((*out)->device,64,"%s",device);
    if(nargs){assert(nargs==2);va_list args;va_start(args,out);assert(!strcmp(va_arg(args,const char*),"INFERENCE_PRECISION_HINT"));const char *value=va_arg(args,const char*);assert(!strcmp(value,"f32")||!strcmp(value,"f16"));va_end(args);}return 0;
}
static void ov_compiled_model_free(CNGOVCompiled *p){free_handle(p);}
static int ov_compiled_model_get_property(const CNGOVCompiled *m,const char *key,char **value){
    if(!strcmp(key,"EXECUTION_DEVICES")){*value=text_copy(fault==F_PLACEMENT?"CPU":strcmp(m->device,"GPU")?m->device:"GPU.0");return 0;}
    if(!strcmp(key,"INFERENCE_PRECISION_HINT")){if(fault==F_PRECISION)return -1;*value=text_copy("f32");return 0;}return -1;
}
static int ov_compiled_model_create_infer_request(const CNGOVCompiled *m,CNGOVRequest **out){*out=alloc_handle(sizeof(**out));(*out)->tokenizer=m->tokenizer;return 0;}
static void ov_tensor_free(CNGOVTensor *t){if(t&&--t->refs==0){free(t->data);free_handle(t);}}
static void ov_infer_request_free(CNGOVRequest *p){if(p){ov_tensor_free(p->in[0]);ov_tensor_free(p->in[1]);free_handle(p);}}
static int ov_tensor_create(int type,CNGShape shape,CNGOVTensor **out){assert(shape.rank>0&&shape.rank<=4);*out=alloc_handle(sizeof(**out));CNGOVTensor *t=*out;t->refs=1;t->type=type;t->rank=shape.rank;t->count=1;for(int i=0;i<shape.rank;i++){t->dims[i]=shape.dims[i];t->count*=shape.dims[i];}t->data=calloc(t->count,type==CNG_OV_I64?8:4);assert(t->data);return 0;}
static int ov_tensor_create_from_string_array(const char **strings,size_t n,CNGShape shape,CNGOVTensor **out){assert(n==1&&shape.rank==1&&shape.dims[0]==1);*out=alloc_handle(sizeof(**out));(*out)->refs=1;(*out)->type=CNG_OV_STRING;(*out)->rank=1;(*out)->dims[0]=1;(*out)->count=1;(*out)->data=text_copy(strings[0]);return 0;}
static int ov_tensor_data(const CNGOVTensor *t,void **out){*out=t->data;return 0;}
static int ov_tensor_get_shape(const CNGOVTensor *t,CNGShape *shape){shape->rank=t->rank;shape->dims=malloc((size_t)t->rank*8);assert(shape->dims);memcpy(shape->dims,t->dims,(size_t)t->rank*8);return 0;}
static int ov_tensor_get_element_type(const CNGOVTensor *t,int *type){*type=t->type;return 0;}
static int ov_tensor_get_size(const CNGOVTensor *t,size_t *n){*n=t->count;return 0;}
static int set_input(CNGOVRequest *r,size_t i,const CNGOVTensor *t){assert(i<2);ov_tensor_free(r->in[i]);r->in[i]=(CNGOVTensor*)t;r->in[i]->refs++;return 0;}
static int ov_infer_request_set_tensor(CNGOVRequest *r,const char *name,const CNGOVTensor *t){return set_input(r,name[0]=='0'?0:1,t);}
static int ov_infer_request_set_input_tensor_by_index(CNGOVRequest *r,size_t i,const CNGOVTensor *t){assert(r->tokenizer);return set_input(r,i,t);}
static int ov_infer_request_get_tensor(const CNGOVRequest *r,const char *name,CNGOVTensor **out){assert(r->tokenizer&&!strcmp(name,"input_ids"));int64_t d[2]={1,token_size};CNGShape s={2,d};return ov_tensor_create(CNG_OV_I64,s,out);}
static int ov_infer_request_get_output_tensor_by_index(const CNGOVRequest *r,size_t i,CNGOVTensor **out){assert(!r->tokenizer&&i==0);int64_t d[4]={1,3,8,12};CNGShape s={4,d};ov_tensor_create(CNG_OV_F32,s,out);for(size_t n=0;n<(*out)->count;n++)((float*)(*out)->data)[n]=((float*)r->in[1]->data)[n]/255.f+.025f;if(fault==F_OUTPUT_NAN)((float*)(*out)->data)[3]=NAN;if(fault==F_OUTPUT_SHAPE)(*out)->dims[2]++;if(fault==F_OUTPUT_TYPE)(*out)->type=3;return 0;}
static int ov_infer_request_infer(CNGOVRequest *r){if(r->tokenizer)token_calls++;else sr_calls++;return fault==F_INFER?-1:0;}
static int ov_genai_llm_pipeline_create(const char *path,const char *device,size_t n,CNGLlm **out,...){assert(path&&n==0);*out=alloc_handle(sizeof(**out));snprintf((*out)->device,64,"%s",device);return 0;}
static void ov_genai_llm_pipeline_free(CNGLlm *p){free_handle(p);}
static int ov_genai_llm_pipeline_get_generation_config(const CNGLlm *p,CNGConfig **out){assert(p);*out=alloc_handle(sizeof(**out));(*out)->tokens=999;(*out)->apply=true;(*out)->model_eos=42;return fault==F_CONFIG?-1:0;}
static int ov_genai_llm_pipeline_render_chat_template(const CNGLlm *p,const char*const*roles,const char*const*contents,size_t n,char*out,size_t*size){assert(p&&n);for(size_t i=0;i<n;i++)assert(roles[i]&&contents[i]);const char *s="<native-template> user prompt <assistant>";size_t need=strlen(s)+1;if(!out){*size=need;return 0;}assert(*size>=need);memcpy(out,s,need);*size=need;if(fault==F_BAD_TEXT)out[0]=(char)0xff;return 0;}
static void ov_genai_generation_config_free(CNGConfig *c){free_handle(c);}
static int ov_genai_generation_config_set_max_new_tokens(CNGConfig *c,size_t n){c->tokens=n;return 0;}
static int ov_genai_generation_config_set_do_sample(CNGConfig *c,bool b){c->sample=b;return 0;}
static int ov_genai_generation_config_set_num_beams(CNGConfig *c,size_t n){c->beams=n;return 0;}
static int ov_genai_generation_config_set_num_return_sequences(CNGConfig *c,size_t n){c->returns=n;return 0;}
static int ov_genai_generation_config_set_apply_chat_template(CNGConfig *c,bool b){c->apply=b;return 0;}
static int ov_genai_llm_pipeline_generate(CNGLlm *p,const char *input,const CNGConfig *c,const void *streamer,CNGResult **out){assert(p&&input&&!streamer&&!c->sample&&c->beams==1&&c->returns==1&&!c->apply);assert(c->tokens<=256&&c->model_eos==42);text_calls++;*out=alloc_handle(sizeof(**out));return 0;}
static int get_text(char *out,size_t *size){const char *s="native result";size_t n=strlen(s)+1;if(!out){*size=n;return 0;}assert(*size>=n);memcpy(out,s,n);if(fault==F_BAD_TEXT)out[0]=(char)0xff;if(fault==F_TEXT_SIZE)*size=n+100;else *size=n;if(fault==F_NO_NUL)out[n-1]='x';return 0;}
static int ov_genai_decoded_results_get_string(const CNGResult *r,char *out,size_t *n){assert(r);return get_text(out,n);}
static void ov_genai_decoded_results_free(CNGResult *r){free_handle(r);}
static int ov_genai_whisper_pipeline_create(const char *path,const char *device,size_t n,CNGWhisper **out,...){assert(path&&n==0);*out=alloc_handle(sizeof(**out));snprintf((*out)->device,64,"%s",device);return 0;}
static void ov_genai_whisper_pipeline_free(CNGWhisper *p){free_handle(p);}
static int ov_genai_whisper_pipeline_get_generation_config(const CNGWhisper*p,CNGWhisperConfig**out){assert(p);*out=alloc_handle(sizeof(**out));(*out)->tokens=999;(*out)->model_eos=42;return 0;}
static int ov_genai_whisper_generation_config_set_max_new_tokens(CNGWhisperConfig*c,size_t n){assert(n<=448);c->tokens=(unsigned)n;return 0;}
static int ov_genai_whisper_generation_config_set_language(CNGWhisperConfig *c,const char *p){assert(!p||!strcmp(p,"<|en|>"));snprintf(c->language,16,"%s",p?p:"auto");return 0;}
static int ov_genai_whisper_generation_config_set_task(CNGWhisperConfig *c,const char *p){snprintf(c->task,16,"%s",p);return 0;}
static void ov_genai_whisper_generation_config_free(CNGWhisperConfig *c){free_handle(c);}
static int ov_genai_whisper_pipeline_generate(CNGWhisper *p,const float *samples,size_t n,const CNGWhisperConfig *config,CNGWhisperResult **out){assert(p&&samples&&n&&n<480000&&config->task[0]&&config->language[0]&&config->model_eos==42&&config->tokens<=448);audio_calls++;*out=alloc_handle(sizeof(**out));return 0;}
static int ov_genai_whisper_decoded_results_get_texts_count(const CNGWhisperResult *r,size_t *n){assert(r);*n=1;return 0;}
static int ov_genai_whisper_decoded_results_get_text_at(const CNGWhisperResult *r,size_t i,char *out,size_t *n){assert(r&&i==0);return get_text(out,n);}
static void ov_genai_whisper_decoded_results_free(CNGWhisperResult *r){free_handle(r);}

static void *fake_open(const char *p,char *error,size_t cap){if(!p||!strcmp(p,"missing")){cng_error(CNG_NO_SDK,error,cap,"missing fake native C SDK");return NULL;}libraries++;return (void*)(uintptr_t)(strstr(p,"genai")?2:1);}
static void fake_close(void *p){if(p)libraries--;}
static void *fake_symbol(void *lib,const char *name){assert(lib);if(missing_symbol&&!strcmp(missing_symbol,name))return NULL;
#define MAP(f) if(!strcmp(name,#f)){void (*fn)(void)=(void(*)(void))f;void *p=NULL;assert(sizeof(fn)==sizeof(p));memcpy(&p,&fn,sizeof(p));return p;}
    MAP(ov_core_create)MAP(ov_core_free)MAP(ov_core_get_available_devices)MAP(ov_available_devices_free)MAP(ov_get_last_err_msg)MAP(ov_free)
    MAP(ov_core_read_model)MAP(ov_model_free)MAP(ov_core_add_extension)MAP(ov_model_inputs_size)MAP(ov_model_outputs_size)
    MAP(ov_model_const_input_by_name)MAP(ov_model_const_output_by_index)MAP(ov_const_port_get_shape)MAP(ov_port_get_element_type)MAP(ov_output_const_port_free)MAP(ov_shape_free)
    MAP(ov_core_compile_model)MAP(ov_compiled_model_free)MAP(ov_compiled_model_get_property)MAP(ov_compiled_model_create_infer_request)MAP(ov_infer_request_free)
    MAP(ov_tensor_create)MAP(ov_tensor_create_from_string_array)MAP(ov_tensor_data)MAP(ov_tensor_get_shape)MAP(ov_tensor_get_element_type)MAP(ov_tensor_get_size)MAP(ov_tensor_free)
    MAP(ov_infer_request_set_tensor)MAP(ov_infer_request_set_input_tensor_by_index)MAP(ov_infer_request_get_tensor)MAP(ov_infer_request_get_output_tensor_by_index)MAP(ov_infer_request_infer)
    MAP(ov_genai_llm_pipeline_create)MAP(ov_genai_llm_pipeline_free)MAP(ov_genai_llm_pipeline_generate)MAP(ov_genai_llm_pipeline_get_generation_config)MAP(ov_genai_llm_pipeline_render_chat_template)
    MAP(ov_genai_generation_config_free)MAP(ov_genai_generation_config_set_max_new_tokens)MAP(ov_genai_generation_config_set_do_sample)MAP(ov_genai_generation_config_set_num_beams)
    MAP(ov_genai_generation_config_set_num_return_sequences)MAP(ov_genai_generation_config_set_apply_chat_template)MAP(ov_genai_decoded_results_get_string)MAP(ov_genai_decoded_results_free)
    MAP(ov_genai_whisper_pipeline_create)MAP(ov_genai_whisper_pipeline_free)MAP(ov_genai_whisper_pipeline_generate)MAP(ov_genai_whisper_pipeline_get_generation_config)MAP(ov_genai_whisper_generation_config_set_max_new_tokens)
    MAP(ov_genai_whisper_generation_config_set_language)MAP(ov_genai_whisper_generation_config_set_task)MAP(ov_genai_whisper_generation_config_free)
    MAP(ov_genai_whisper_decoded_results_get_texts_count)MAP(ov_genai_whisper_decoded_results_get_text_at)MAP(ov_genai_whisper_decoded_results_free)
#undef MAP
    return NULL;
}
static void clean(void){assert(handles==0&&libraries==0);fault=0;missing_symbol=NULL;}
static CNGEngine *engine(const char *model,CNGModality modality,const char *device){
    CNGOptions o={.model_dir=model,.modality=modality,.device=device,.ov_c_library="core",.genai_c_library="/sdk/genai",.tokenizers_library="tokenizers",.sr_xml="trained.xml"};CNGEngine *e=NULL;char err[256];assert(cng_create(&o,&e,err,sizeof(err))==CNG_OK&&e);return e;
}
int main(void){
    const char *dir="verified-model";char error[256];
    char *output=NULL;size_t n=987;assert(cng_available_devices("core",&output,&n,error,sizeof(error))==CNG_OK);assert(!strcmp(output,"CPU\nGPU.0\nGPU.1\n"));cng_free(output);output=NULL;clean();
    assert(cng_available_devices("missing",&output,&n,error,sizeof(error))==CNG_NO_SDK&&!output);clean();
    fault=F_DEVICE_LIST;assert(cng_available_devices("core",&output,&n,error,sizeof(error))==CNG_SDK_ERROR&&!output);clean();
    CNGOptions o={.model_dir=dir,.modality=CNG_TEXT,.device="AUTO",.ov_c_library="core",.genai_c_library="/sdk/genai",.tokenizers_library="tokenizers"};CNGEngine *e=NULL;
    assert(cng_create(&o,&e,error,sizeof(error))==CNG_INVALID&&!e);clean();o.device="NPU";assert(cng_create(&o,&e,error,sizeof(error))==CNG_DEVICE&&!e);clean();
    o.device="GPU";missing_symbol="ov_genai_generation_config_set_apply_chat_template";assert(cng_create(&o,&e,error,sizeof(error))==CNG_NO_SDK&&!e&&strstr(error,"set_apply_chat_template"));clean();
    e=engine(dir,CNG_TEXT,"GPU");CNGInfo info;assert(cng_info(e,&info)==CNG_OK&&!info.execution_devices_verified&&!strcmp(info.selected_device,"GPU"));
    const char *roles[2]={"system","user"},*contents[2]={"instructions","user prompt"};
    assert(cng_render_chat(e,roles,contents,2,&output,&n,error,sizeof(error))==CNG_OK&&strstr(output,"<native-template>"));cng_free(output);output=NULL;
    roles[1]="tool";n=123;assert(cng_render_chat(e,roles,contents,2,&output,&n,error,sizeof(error))==CNG_INVALID&&!output&&n==123);roles[1]="user";
    contents[1]="\xff";assert(cng_render_chat(e,roles,contents,2,&output,&n,error,sizeof(error))==CNG_INVALID&&!output);contents[1]="user prompt";
    assert(cng_render_chat(e,roles,contents,65,&output,&n,error,sizeof(error))==CNG_INVALID);
    fault=F_BAD_TEXT;assert(cng_render_chat(e,roles,contents,2,&output,&n,error,sizeof(error))==CNG_SDK_ERROR&&!output);fault=0;
    assert(cng_text(e,"raw prompt",10,32,&output,&n,error,sizeof(error))==CNG_OK&&!strcmp(output,"native result"));size_t count=0;assert(cng_last_prompt_tokens(e,&count)==CNG_OK&&count==7);cng_free(output);output=NULL;
    int calls=text_calls;token_size=2049;assert(cng_text(e,"raw prompt",10,32,&output,&n,error,sizeof(error))==CNG_LIMIT&&!output&&text_calls==calls);token_size=7;
    assert(cng_text(e,"\xff",1,32,&output,&n,error,sizeof(error))==CNG_INVALID);assert(cng_text(e,"x\0y",3,32,&output,&n,error,sizeof(error))==CNG_INVALID);
    assert(cng_text(e,"prompt",6,257,&output,&n,error,sizeof(error))==CNG_LIMIT);
    int broken[]={F_BAD_TEXT,F_TEXT_SIZE,F_NO_NUL,F_INFER,F_CONFIG};for(size_t i=0;i<sizeof(broken)/sizeof(broken[0]);i++){fault=broken[i];n=123;assert(cng_text(e,"raw prompt",10,32,&output,&n,error,sizeof(error))!=CNG_OK&&!output&&n==123);}fault=0;
    cng_destroy(e);e=NULL;clean();
    o.max_prompt_tokens=12;o.max_context_tokens=15;o.max_tokens=8;assert(cng_create(&o,&e,error,sizeof(error))==CNG_OK);n=123;assert(cng_text(e,"raw prompt",10,9,&output,&n,error,sizeof(error))==CNG_LIMIT);token_size=8;assert(cng_text(e,"raw prompt",10,8,&output,&n,error,sizeof(error))==CNG_LIMIT);token_size=7;cng_destroy(e);e=NULL;clean();
    e=engine(dir,CNG_AUDIO,"CPU");float samples[16]={0};assert(cng_whisper(e,samples,16,"transcribe","en",&output,&n,error,sizeof(error))==CNG_OK);cng_free(output);output=NULL;
    assert(cng_whisper(e,samples,16,"translate",NULL,&output,&n,error,sizeof(error))==CNG_OK);cng_free(output);output=NULL;
    samples[2]=NAN;assert(cng_whisper(e,samples,16,"transcribe","en",&output,&n,error,sizeof(error))==CNG_NONFINITE);samples[2]=0;
    samples[2]=1.01f;assert(cng_whisper(e,samples,16,"transcribe","en",&output,&n,error,sizeof(error))==CNG_INVALID);samples[2]=-1.01f;assert(cng_whisper(e,samples,16,"translate",NULL,&output,&n,error,sizeof(error))==CNG_INVALID);samples[2]=0;
    assert(cng_whisper(e,samples,480000,"transcribe","en",&output,&n,error,sizeof(error))==CNG_INVALID);assert(cng_whisper(e,samples,16,"other","en",&output,&n,error,sizeof(error))==CNG_INVALID);assert(cng_whisper(e,samples,16,"translate","fr",&output,&n,error,sizeof(error))==CNG_INVALID);
    cng_destroy(e);e=NULL;clean();
    CNGOptions sr={.model_dir=dir,.modality=CNG_CAMERA,.device="GPU",.ov_c_library="core",.sr_xml="trained.xml"};fault=F_PLACEMENT;assert(cng_create(&sr,&e,error,sizeof(error))==CNG_DEVICE&&!e);clean();fault=F_META;assert(cng_create(&sr,&e,error,sizeof(error))==CNG_INVALID&&!e);clean();
    e=engine(dir,CNG_CAMERA,"GPU");assert(cng_info(e,&info)==CNG_OK&&info.execution_devices_verified&&!strcmp(info.selected_device,"GPU.0")&&info.scale==4&&info.lr_width==3&&info.lr_height==2);
    float lr[18],bc[288],*pixels=NULL;for(int i=0;i<18;i++)lr[i]=127.f;for(int i=0;i<288;i++)bc[i]=63.f;
    assert(cng_sr(e,lr,18,bc,288,&pixels,&n,error,sizeof(error))==CNG_OK&&n==288&&fabsf(pixels[0]-(63.f/255.f+.025f))<1e-6);cng_free(pixels);pixels=NULL;
    int sf[]={F_OUTPUT_NAN,F_OUTPUT_SHAPE,F_OUTPUT_TYPE,F_INFER,F_PLACEMENT};for(size_t i=0;i<sizeof(sf)/sizeof(sf[0]);i++){fault=sf[i];n=123;assert(cng_sr(e,lr,18,bc,288,&pixels,&n,error,sizeof(error))!=CNG_OK&&!pixels&&n==123);}fault=0;
    assert(cng_sr(e,lr,17,bc,288,&pixels,&n,error,sizeof(error))==CNG_INVALID);lr[0]=256;assert(cng_sr(e,lr,18,bc,288,&pixels,&n,error,sizeof(error))==CNG_NONFINITE);lr[0]=127;
    cng_destroy(e);e=NULL;clean();
    sr.device="GPU.1";assert(cng_create(&sr,&e,error,sizeof(error))==CNG_OK);assert(cng_info(e,&info)==CNG_OK&&!strcmp(info.selected_device,"GPU.1"));cng_destroy(e);e=NULL;clean();
    printf("native C backend fake-SDK: text%d/audio%d/SR%d/tokenizer%d calls, limits, device refusal, exact ownership passed\n",text_calls,audio_calls,sr_calls,token_calls);return 0;
}
