/* Native Mojo/MAX tensor inference and runtime-visible device probe. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "native_mojo_backend.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void quoted(const char *s) {
    putchar('"');for(const unsigned char *p=(const unsigned char*)s;*p;p++) {
        if(*p=='"'||*p=='\\'){putchar('\\');putchar(*p);}
        else if(*p<32)printf("\\u%04x",*p);else putchar(*p);
    }putchar('"');
}
static unsigned char *read_file(const char *path,size_t limit,size_t *bytes) {
    struct stat st;if(lstat(path,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||(uint64_t)st.st_size>limit)return NULL;
    int fd=open(path,O_RDONLY|O_NONBLOCK|O_NOFOLLOW);if(fd<0)return NULL;
    struct stat now;if(fstat(fd,&now)||st.st_ino!=now.st_ino||st.st_dev!=now.st_dev||st.st_size!=now.st_size){close(fd);return NULL;}
    unsigned char *data=malloc((size_t)st.st_size+1);if(!data){close(fd);return NULL;}
    size_t at=0;
    while(at<(size_t)st.st_size){ssize_t n=read(fd,data+at,(size_t)st.st_size-at);if(n<=0){free(data);close(fd);return NULL;}at+=(size_t)n;}
    unsigned char extra;if(read(fd,&extra,1)!=0){free(data);close(fd);return NULL;}
    struct stat after;
    if(fstat(fd,&after)||after.st_size!=now.st_size){free(data);close(fd);return NULL;}
#ifdef __APPLE__
    if(after.st_mtimespec.tv_sec!=now.st_mtimespec.tv_sec||after.st_mtimespec.tv_nsec!=now.st_mtimespec.tv_nsec){free(data);close(fd);return NULL;}
#else
    if(after.st_mtim.tv_sec!=now.st_mtim.tv_sec||after.st_mtim.tv_nsec!=now.st_mtim.tv_nsec){free(data);close(fd);return NULL;}
#endif
    close(fd);data[at]=0;*bytes=at;return data;
}
static const char *str(jval *v,const char *key){jval *s=json_get(v,key);return s&&s->t==J_STR?s->str:NULL;}
static int unique(jval *v) {
    if(!v)return 0;
    if(v->t==J_OBJ)for(int a=0;a<v->len;a++){for(int b=0;b<a;b++)if(!strcmp(v->keys[a],v->keys[b]))return 0;if(!unique(v->kids[a]))return 0;}
    if(v->t==J_ARR)for(int a=0;a<v->len;a++)if(!unique(v->kids[a]))return 0;
    return v->t!=J_NUM||isfinite(v->num);
}
static CMJDType dtype(const char *s) {
    static const char *names[]={"","f32","i8","i16","i4","i64","i32","u8","f16","bf16"};
    if(s)for(unsigned k=1;k<sizeof names/sizeof names[0];k++)if(!strcmp(s,names[k]))return (CMJDType)k;
    return (CMJDType)0;
}
static const char *dtype_name(CMJDType d) {
    static const char *names[]={"unknown","f32","i8","i16","i4","i64","i32","u8","f16","bf16"};
    return d>=CMJ_F32&&d<=CMJ_BF16?names[d]:names[0];
}
static int positive(const char *s,unsigned *out) {
    if(!s||!*s)return 0;unsigned n=0;
    for(;*s;s++){if(*s<'0'||*s>'9'||n>(10000u-(unsigned)(*s-'0'))/10)return 0;n=n*10+(unsigned)(*s-'0');}
    *out=n;return n>0;
}
static double elapsed(struct timespec a,struct timespec b){return (double)(b.tv_sec-a.tv_sec)+(double)(b.tv_nsec-a.tv_nsec)/1e9;}
int main(int argc,char **argv) {
#ifdef COLI_MOJO_CONFIG_DIR
    const char *sdk_config=getenv("COLI_MOJO_CONFIG");
    char adjacent[4096],cfg_file[4096];
    if(!sdk_config) {
        int n=snprintf(adjacent,sizeof adjacent,"%s.sdk",argv[0]);
        int m=n>0&&(size_t)n<sizeof adjacent?snprintf(cfg_file,sizeof cfg_file,"%s/modular.cfg",adjacent):-1;
        struct stat cfg;
        sdk_config=m>0&&(size_t)m<sizeof cfg_file&&!stat(cfg_file,&cfg)&&S_ISREG(cfg.st_mode)?adjacent:COLI_MOJO_CONFIG_DIR;
    }
    setenv("MODULAR_HOME",sdk_config,1);
#endif
    const char *model=NULL,*manifest=NULL,*output=NULL,*device="CPU";unsigned repeats=1;int probe=0;
    const char *weights[32];size_t weight_count=0;
    for(int k=1;k<argc;k++) {
        if(!strcmp(argv[k],"--probe")){probe=1;continue;}
        if(k+1>=argc)goto usage;const char *key=argv[k++],*value=argv[k];
        if(!strcmp(key,"--model"))model=value;else if(!strcmp(key,"--inputs"))manifest=value;
        else if(!strcmp(key,"--output"))output=value;else if(!strcmp(key,"--device"))device=value;
        else if(!strcmp(key,"--weights")){if(weight_count==32)goto usage;weights[weight_count++]=value;}
        else if(!strcmp(key,"--repeat")){if(!positive(value,&repeats))goto usage;}else goto usage;
    }
    int accelerator=-1;
    if(strcmp(device,"CPU")) {
        const char *id=!strncmp(device,"ACCELERATOR.",12)?device+12:NULL;
        if(!id||!*id)goto usage;unsigned value=0;
        for(;*id;id++){if(*id<'0'||*id>'9'||value>1000)goto usage;value=value*10+(unsigned)(*id-'0');}
        if(value>1000)goto usage;accelerator=(int)value;
    }
    if(probe) {
        int count=cmj_accelerator_count(),host=cmj_host_available();
        fputs("{\"schema\":\"colibri.mojo-device-probe/v1\",\"sdk_version\":",stdout);quoted(cmj_version());
        printf(",\"cpu_available\":%s,\"accelerator_count\":%d,\"enumeration_scope\":\"Modular runtime-visible devices\",\"npu_tpu_aipu\":\"requires compatible driver/plugin; no universal discovery claim\",\"native_windows\":false}\n",host?"true":"false",count);
        return count<0||!host?2:0;
    }
    if(!model||!manifest||!output)goto usage;
    CMJInput in[CMJ_MAX_TENSORS]={0};CMJOutput out[CMJ_MAX_TENSORS]={0};
    struct timespec reading,published,cleaned;clock_gettime(CLOCK_MONOTONIC,&reading);
    void *payloads[CMJ_MAX_TENSORS]={0};const char *names[CMJ_MAX_TENSORS]={0};
    CMJEngine *engine=NULL;jval *root=NULL;unsigned char *json=NULL;size_t size=0,count=0,names_count=0,total=0;int result=2;
    char error[4096]="invalid native input manifest";
    json=read_file(manifest,65536,&size);if(!json||memchr(json,0,size))goto done;
    /* Reject decoded NUL escapes; literal backslash-u text is permitted. */
    for(size_t k=0;k<size;k++)if(json[k]=='\\'&&k+1<size){if(json[k+1]=='u'&&k+5<size&&!memcmp(json+k+2,"0000",4))goto done;k++;}
    root=json_parse_checked((const char*)json);if(!root||root->t!=J_OBJ||!unique(root))goto done;
    jval *inputs=json_get(root,"inputs"),*outputs=json_get(root,"outputs");
    if(!inputs||inputs->t!=J_ARR||inputs->len<1||(unsigned)inputs->len>CMJ_MAX_TENSORS||
       !outputs||outputs->t!=J_ARR||outputs->len<1||(unsigned)outputs->len>CMJ_MAX_TENSORS)goto done;
    count=(size_t)inputs->len;names_count=(size_t)outputs->len;
    for(size_t k=0;k<count;k++) {
        jval *entry=inputs->kids[k],*shape=json_get(entry,"shape");const char *file=str(entry,"file");
        in[k].name=str(entry,"name");in[k].dtype=dtype(str(entry,"dtype"));
        if(!file||!in[k].name||!in[k].dtype||!shape||shape->t!=J_ARR||(unsigned)shape->len>CMJ_MAX_RANK)goto done;
        in[k].rank=(size_t)shape->len;
        for(size_t j=0;j<in[k].rank;j++){jval *d=shape->kids[j];if(d->t!=J_NUM||d->num<1||d->num>2147483647||floor(d->num)!=d->num)goto done;in[k].shape[j]=(int64_t)d->num;}
        char resolved[4096];const char *input_path=file;
        if(file[0]!='/') {
            const char *slash=strrchr(manifest,'/');size_t prefix=slash?(size_t)(slash-manifest)+1:0;
            if(prefix+strlen(file)>=sizeof resolved)goto done;
            memcpy(resolved,manifest,prefix);strcpy(resolved+prefix,file);input_path=resolved;
        }
        payloads[k]=read_file(input_path,CMJ_MAX_BYTES-total,&in[k].bytes);if(!payloads[k])goto done;
        in[k].data=payloads[k];total+=in[k].bytes;
    }
    for(size_t k=0;k<names_count;k++){if(outputs->kids[k]->t!=J_STR)goto done;names[k]=outputs->kids[k]->str;}
    CMJOptions options={accelerator,model,CMJ_MAX_BYTES,weights,weight_count};
    struct timespec begin,loaded,finished;clock_gettime(CLOCK_MONOTONIC,&begin);
    if(cmj_create(&options,&engine,error,sizeof error)!=CMJ_OK)goto done;
    clock_gettime(CLOCK_MONOTONIC,&loaded);
    for(unsigned k=0;k<repeats;k++) {
        cmj_output_free(out,names_count);
        if(cmj_run(engine,in,count,names,names_count,out,error,sizeof error)!=CMJ_OK)goto done;
    }
    clock_gettime(CLOCK_MONOTONIC,&finished);
    snprintf(error,sizeof error,"native output publication failed; partial artifacts may remain");
    if(mkdir(output,0700)){snprintf(error,sizeof error,"output directory must be new and writable");goto done;}
    /* Write exclusive files only after native inference succeeds. */
    for(size_t k=0;k<names_count;k++) {
        char path[4096];int n=snprintf(path,sizeof path,"%s/output-%zu.bin",output,k);
        if(n<0||(size_t)n>=sizeof path)goto done;
        int fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)goto done;
        size_t at=0;while(at<out[k].bytes){ssize_t w=write(fd,(unsigned char*)out[k].data+at,out[k].bytes-at);if(w<=0)break;at+=(size_t)w;}
        int closed=close(fd);if(at!=out[k].bytes||closed)goto done;
    }
    clock_gettime(CLOCK_MONOTONIC,&published);
    cmj_destroy(engine);engine=NULL;
    for(size_t k=0;k<names_count;k++){free(out[k].data);out[k].data=NULL;}
    for(size_t k=0;k<count;k++){free(payloads[k]);payloads[k]=NULL;}
    json_free(root);root=NULL;free(json);json=NULL;
    clock_gettime(CLOCK_MONOTONIC,&cleaned);
    fputs("{\"schema\":\"colibri.mojo-native-run/v1\",\"backend\":\"modular-max-native-c\",\"sdk_version\":",stdout);quoted(cmj_version());
    fputs(",\"requested_device\":",stdout);quoted(device);
    printf(",\"inference_runtime\":\"native-c\",\"python_model_bindings\":false,\"device_proof\":\"checked source output tensor placement; not all-node proof\",\"input_read_seconds\":%.9f,\"compile_load_seconds\":%.9f,\"run_seconds\":%.9f,\"publish_seconds\":%.9f,\"cleanup_seconds\":%.9f,\"total_seconds\":%.9f,\"repeats\":%u,\"outputs\":[",elapsed(reading,begin),elapsed(begin,loaded),elapsed(loaded,finished),elapsed(finished,published),elapsed(published,cleaned),elapsed(reading,cleaned),repeats);
    for(size_t k=0;k<names_count;k++) {
        if(k)putchar(',');fputs("{\"name\":",stdout);quoted(out[k].name);
        printf(",\"dtype\":\"%s\",\"bytes\":%zu,\"source_accelerator\":%s,\"source_device_id\":%d,\"shape\":[",
               dtype_name(out[k].dtype),out[k].bytes,out[k].source_accelerator?"true":"false",out[k].source_device_id);
        for(size_t j=0;j<out[k].rank;j++){if(j)putchar(',');printf("%lld",(long long)out[k].shape[j]);}
        fputs("]}",stdout);
    }puts("]}");result=0;
done:
    if(result)fprintf(stderr,"mojo_runtime: %s\n",error);
    cmj_output_free(out,names_count);cmj_destroy(engine);for(size_t k=0;k<CMJ_MAX_TENSORS;k++)free(payloads[k]);
    json_free(root);free(json);return result;
usage:
    fputs("mojo_runtime --probe | --model graph.mef --inputs request.json --output new-dir [--device CPU|ACCELERATOR.id] [--repeat N] [--weights shard.safetensors ...]\n",stderr);return 2;
}
