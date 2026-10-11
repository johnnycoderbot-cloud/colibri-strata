/* Persistent C inference engine: raw transport, native SDK calls and native
 * tensor preparation. Python control-plane code never executes the model. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "native_graph_backend.h"
#include "native_graph_manifest.h"
#include "native_media.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#endif
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 1048576
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include "third_party/stb_image_write.h"

typedef struct { unsigned char *p;size_t n,cap,limit;int failed; } Bytes;
static int append(Bytes *b,const void *p,size_t n) {
    if(b->failed||n>b->limit-b->n){b->failed=1;return 0;}
    if(b->n+n>b->cap){size_t cap=b->cap?b->cap:1024;while(cap<b->n+n){if(cap>b->limit/2){cap=b->limit;break;}cap*=2;}
        void *q=realloc(b->p,cap);if(!q){b->failed=1;return 0;}b->p=q;b->cap=cap;}
    if(n)memcpy(b->p+b->n,p,n);b->n+=n;return 1;
}
static void png_write(void *p,void *data,int n){if(n<0)((Bytes*)p)->failed=1;else append(p,data,(size_t)n);}
static uint32_t be32(const unsigned char *p){return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];}
static uint32_t crc_bytes(const unsigned char *p,size_t n){uint32_t c=~0u;for(size_t i=0;i<n;i++){c^=p[i];for(int j=0;j<8;j++)c=(c>>1)^(0xedb88320u&(0u-(c&1u)));}return ~c;}
static int png_decode(const unsigned char *p,size_t bytes,size_t max_pixels,unsigned char **rgba,size_t *width,size_t *height,int *alpha,char *err,size_t cap) {
    const unsigned char sig[]={137,80,78,71,13,10,26,10};size_t pos=8,w=0,h=0;int idat=0,end=0,has_alpha=0;
    if(!p||bytes<45||bytes>((size_t)32<<20)||memcmp(p,sig,8))goto invalid;
    while(pos<bytes) {
        if(bytes-pos<12)goto invalid;size_t n=be32(p+pos);
        if(n>bytes-pos-12)goto invalid;
        const unsigned char *kind=p+pos+4,*data=p+pos+8;
        for(int j=0;j<4;j++)if(!((kind[j]>='A'&&kind[j]<='Z')||(kind[j]>='a'&&kind[j]<='z')))goto invalid;
        if(crc_bytes(kind,n+4)!=be32(data+n))goto invalid;
        if(!w) {
            if(memcmp(kind,"IHDR",4)||n!=13)goto invalid;w=be32(data);h=be32(data+4);
            if(!w||!h||w>1048576||h>1048576||w>max_pixels/h||w*h>max_pixels/16||data[8]!=8||
               !(data[9]==0||data[9]==2||data[9]==3||data[9]==4||data[9]==6)||data[10]||data[11]||data[12]>1)goto invalid;
            has_alpha=data[9]==4||data[9]==6;
        } else if(!memcmp(kind,"IHDR",4))goto invalid;
        if(!memcmp(kind,"IDAT",4))idat=1;
        else if(!memcmp(kind,"IEND",4)){if(n||!idat||pos+12!=bytes)goto invalid;end=1;}
        else if(!memcmp(kind,"tRNS",4))has_alpha=1;
        else if(kind[0]>='A'&&kind[0]<='Z'&&memcmp(kind,"IHDR",4)&&memcmp(kind,"PLTE",4))goto invalid;
        pos+=n+12;
    }
    if(!end)goto invalid;
    /* STB's malloc inflater otherwise accepts arbitrarily large decoded IDAT
     * padding despite a tiny IHDR. Validate into exactly the scanline budget
     * first, including Adam7 passes. The identical subsequent stream cannot
     * inflate beyond this checked size. */
    unsigned color=p[25],components=color==2?3:color==4?2:color==6?4:1;
    size_t expected=0;
    if(!p[28])expected=h*(w*components+1);
    else {
        static const unsigned sx[]={0,4,0,2,0,1,0},sy[]={0,0,4,0,2,0,1};
        static const unsigned dx[]={8,8,4,4,2,2,1},dy[]={8,8,8,4,4,2,2};
        for(size_t k=0;k<7;k++){size_t pw=w>sx[k]?(w-sx[k]+dx[k]-1)/dx[k]:0,ph=h>sy[k]?(h-sy[k]+dy[k]-1)/dy[k]:0;if(pw&&ph)expected+=ph*(pw*components+1);}
    }
    unsigned char *compressed=malloc(bytes),*scanlines=malloc(expected);
    if(!compressed||!scanlines){free(compressed);free(scanlines);goto invalid;}
    size_t compressed_n=0;
    for(size_t at=8;at<bytes;){size_t n=be32(p+at);if(!memcmp(p+at+4,"IDAT",4)){memcpy(compressed+compressed_n,p+at+8,n);compressed_n+=n;}at+=n+12;}
    int expanded=stbi_zlib_decode_buffer((char*)scanlines,(int)expected,(const char*)compressed,(int)compressed_n);
    free(compressed);free(scanlines);if(expanded<0||(size_t)expanded!=expected)goto invalid;
    int x=0,y=0,channels=0;unsigned char *decoded=stbi_load_from_memory(p,(int)bytes,&x,&y,&channels,4);
    if(!decoded||(size_t)x!=w||(size_t)y!=h){stbi_image_free(decoded);goto invalid;}
    *rgba=decoded;*width=w;*height=h;if(alpha)*alpha=has_alpha;return 1;
invalid:snprintf(err,cap,"invalid, truncated or oversized PNG pixels");return 0;
}
static int utf8(const unsigned char *s,size_t n) {
    for(size_t i=0;i<n;) {
        unsigned c=s[i++],cp=0,need=0,min=0;
        if(!c)return 0;if(c<128)continue;
        if(c>=0xc2&&c<=0xdf){cp=c&31;need=1;min=128;}
        else if(c>=0xe0&&c<=0xef){cp=c&15;need=2;min=2048;}
        else if(c>=0xf0&&c<=0xf4){cp=c&7;need=3;min=65536;}else return 0;
        if(need>n-i)return 0;
        while(need--){c=s[i++];if((c&0xc0)!=0x80)return 0;cp=(cp<<6)|(c&63);}
        if(cp<min||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))return 0;
    }return 1;
}
static void quoted(FILE *f,const char *s){fputc('"',f);for(const unsigned char *p=(const unsigned char*)s;*p;p++){
    if(*p=='"'||*p=='\\'){fputc('\\',f);fputc(*p,f);}else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);}fputc('"',f);}
static void ready(const CGManifest *m,const CNGInfo *i) {
    fputs("READY {\"device\":",stdout);quoted(stdout,i->selected_device);fputs(",\"compiled_device\":",stdout);quoted(stdout,i->selected_device);
    fprintf(stdout,",\"execution_devices_verified\":%s,\"inference_runtime\":\"native-c\",\"backend\":\"openvino-native-c\",\"runtime_precision\":",i->execution_devices_verified?"true":"false");quoted(stdout,i->runtime_precision);
    fputs(",\"model_id\":",stdout);quoted(stdout,m->model_id);fputs(",\"family\":",stdout);quoted(stdout,m->family);
    fprintf(stdout,",\"lr_width\":%zu,\"lr_height\":%zu,\"scale\":%u,\"prompt_token_limit\":%u,\"context_token_limit\":%u,\"memory_estimates_only\":true,\"base_estimated_bytes\":%llu}\n",
        i->lr_width,i->lr_height,i->scale,m->max_prompt_tokens,m->max_context_tokens,(unsigned long long)m->base_estimated_bytes);fflush(stdout);
}
static int frame(const char *id,const char *json,size_t bytes,const void *body,size_t n) {
    if(fprintf(stdout,"RESULT %s %zu %zu\n",id,bytes,n)<0||fwrite(json,1,bytes,stdout)!=bytes||fputc('\n',stdout)==EOF||
       (n&&fwrite(body,1,n,stdout)!=n)||fputc('\n',stdout)==EOF)return 0;return fflush(stdout)==0;
}
static int error_frame(const char *id,const char *message) {
    char out[4096];size_t at=10;memcpy(out,"{\"error\":\"",at);
    for(const unsigned char *p=(const unsigned char*)message;*p&&at+8<sizeof out;p++) {
        if(*p=='"'||*p=='\\'){out[at++]='\\';out[at++]=(char)*p;}
        else if(*p<32)at+=(size_t)snprintf(out+at,sizeof out-at,"\\u%04x",*p);else out[at++]=(char)*p;
    }out[at++]='"';out[at++]='}';return frame(id,out,at,NULL,0);
}
static int text_frame(const char *id,const char *text,size_t bytes,const CNGInfo *info,size_t prompt_tokens) {
    Bytes b={0};b.limit=65536;append(&b,"{\"text\":\"",9);
    for(size_t i=0;i<bytes;i++) {unsigned char c=(unsigned char)text[i];char tmp[8];
        if(c=='"'||c=='\\'){tmp[0]='\\';tmp[1]=(char)c;append(&b,tmp,2);}else if(c<32){int n=snprintf(tmp,sizeof tmp,"\\u%04x",c);append(&b,tmp,(size_t)n);}else append(&b,&c,1);}
    char tail[512];int n=snprintf(tail,sizeof tail,"\",\"prompt_token_count\":%zu,\"stateful_chat\":false,\"segments\":[],\"inference_runtime\":\"native-c\",\"device\":\"%s\",\"execution_devices_verified\":%s}",prompt_tokens,info->selected_device,info->execution_devices_verified?"true":"false");
    if(n<0||(size_t)n>=sizeof tail||!append(&b,tail,(size_t)n)){free(b.p);return error_frame(id,"native text result exceeds its bound");}
    int ok=frame(id,(const char*)b.p,b.n,NULL,0);free(b.p);return ok;
}
static int sr_run(CNGEngine *engine,const CGManifest *m,const CNGInfo *i,const unsigned char *payload,size_t bytes,Bytes *png,size_t *outw,size_t *outh,char *error,size_t cap) {
    unsigned char *source=NULL,*result=NULL,*tile_rgb=NULL,*alpha=NULL,*alpha4=NULL,*joined=NULL;float *low=NULL,*high=NULL,*scores=NULL;void *scratch=NULL,*alpha_scratch=NULL;
    size_t w=0,h=0,tw=i->lr_width,th=i->lr_height,work=0,score_count=0;int ok=0,has_alpha=0;
    if(!png_decode(payload,bytes,m->max_output_pixels,&source,&w,&h,&has_alpha,error,cap))return 0;
    size_t ow=w*4,oh=h*4,minimum=tw<th?tw:th;int halo=m->tile_overlap<0?(int)((minimum-1)/2<16?(minimum-1)/2:16):m->tile_overlap;
    if(!tw||!th||tw>1024||th>1024||tw*th>NM_SR_MAX_TILE_PIXELS||halo<0||(size_t)halo>(minimum-1)/2) {snprintf(error,cap,"invalid trained tile geometry or overlap");goto done;}
    size_t lp=tw*th,hp=lp*16,stepw=tw-(size_t)halo*2,steph=th-(size_t)halo*2;
    if(nm_sr_workspace_bytes(tw,th,&work)!=NM_OK){snprintf(error,cap,"invalid media workspace");goto done;}
    result=malloc(ow*oh*3);low=malloc(lp*3*sizeof(float));high=malloc(hp*3*sizeof(float));tile_rgb=malloc(hp*3);scratch=malloc(work);
    if(!result||!low||!high||!tile_rgb||!scratch){snprintf(error,cap,"native image workspace allocation failed");goto done;}
    for(size_t top=0;top<h;top+=steph)for(size_t left=0;left<w;left+=stepw) {
        if(nm_sr_prepare(source,w*h*4,w,h,4,w*4,(int64_t)left-halo,(int64_t)top-halo,tw,th,low,lp*3,high,hp*3,scratch,work)!=NM_OK){snprintf(error,cap,"native model input preparation failed");goto done;}
        if(cng_sr(engine,low,lp*3,high,hp*3,&scores,&score_count,error,cap)!=CNG_OK)goto done;
        if(nm_bgr_output_rgb(scores,score_count,tw*4,th*4,tile_rgb,hp*3,tw*12)!=NM_OK){snprintf(error,cap,"native trained model returned invalid pixels");goto done;}
        cng_free(scores);scores=NULL;size_t takew=w-left<stepw?w-left:stepw,takeh=h-top<steph?h-top:steph;
        for(size_t y=0;y<takeh*4;y++)memcpy(result+((top*4+y)*ow+left*4)*3,tile_rgb+((y+(size_t)halo*4)*tw*4+(size_t)halo*4)*3,takew*12);
    }
    int channels=3;const unsigned char *pixels=result;
    if(has_alpha) {
        size_t aw=0;alpha=malloc(w*h);alpha4=malloc(ow*oh);joined=malloc(ow*oh*4);
        if(nm_bicubic4_workspace_bytes(w,h,1,&aw)!=NM_OK){snprintf(error,cap,"invalid native alpha workspace");goto done;}alpha_scratch=malloc(aw);
        if(!alpha||!alpha4||!joined||!alpha_scratch){snprintf(error,cap,"native alpha allocation failed");goto done;}
        for(size_t k=0;k<w*h;k++)alpha[k]=source[k*4+3];
        if(nm_bicubic4_u8(alpha,w*h,w,h,1,w,alpha4,ow*oh,ow,alpha_scratch,aw)!=NM_OK){snprintf(error,cap,"native alpha resizing failed");goto done;}
        for(size_t k=0;k<ow*oh;k++){memcpy(joined+k*4,result+k*3,3);joined[k*4+3]=alpha4[k];}channels=4;pixels=joined;
    }
    png->limit=m->max_output_bytes;
    if(!stbi_write_png_to_func(png_write,png,(int)ow,(int)oh,channels,pixels,(int)(ow*(size_t)channels))||png->failed){snprintf(error,cap,"native PNG output exceeds byte limit");goto done;}
    *outw=ow;*outh=oh;ok=1;
done:stbi_image_free(source);free(result);free(tile_rgb);free(alpha);free(alpha4);free(joined);free(low);free(high);cng_free(scores);free(scratch);free(alpha_scratch);return ok;
}
static const char *jstr(jval *o,const char *key){jval *v=json_get(o,key);return v&&v->t==J_STR?v->str:NULL;}
static int jint(jval *o,const char *key,unsigned def,unsigned max,unsigned *out){jval *v=json_get(o,key);if(!v){*out=def;return 1;}if(v->t!=J_NUM||!isfinite(v->num)||v->num<1||v->num>max||floor(v->num)!=v->num)return 0;*out=(unsigned)v->num;return 1;}
static int read_exact(void *p,size_t bytes){return !bytes||fread(p,1,bytes,stdin)==bytes;}
static int nul_escape(const char *s){for(size_t i=0;s[i];i++)if(s[i]=='\\'&&s[i+1]){if(s[i+1]=='u'&&!strncmp(s+i+2,"0000",4))return 1;i++;}return 0;}
static int render_prompt(CNGEngine *engine,const unsigned char *body,size_t bytes,int chat,
                         size_t limit,char **rendered,size_t *rendered_bytes,char *error,size_t cap) {
    const char *roles[64],*contents[64];size_t count=1,total=0;jval *messages=NULL;
    if(!utf8(body,bytes)){snprintf(error,cap,"native prompt must be NUL-free UTF8");return 0;}
    if(!chat){if(bytes>limit){snprintf(error,cap,"prompt exceeds byte bound");return 0;}roles[0]="user";contents[0]=(const char*)body;}
    else {
        if(nul_escape((const char*)body)||(messages=json_parse_checked((const char*)body))==NULL||messages->t!=J_ARR||messages->len<1||messages->len>64)goto invalid;
        count=(size_t)messages->len;
        for(size_t k=0;k<count;k++) {
            jval *item=messages->kids[k];
            if(!item||item->t!=J_OBJ||item->len!=2||!strcmp(item->keys[0],item->keys[1]))goto invalid;
            roles[k]=jstr(item,"role");contents[k]=jstr(item,"content");
            if(!roles[k]||!contents[k]||!utf8((const unsigned char*)contents[k],strlen(contents[k]))||strlen(contents[k])>limit-total)goto invalid;
            total+=strlen(contents[k]);
        }
    }
    CNGStatus status=cng_render_chat(engine,roles,contents,count,rendered,rendered_bytes,error,cap);
    json_free(messages);return status==CNG_OK;
invalid:json_free(messages);snprintf(error,cap,"invalid bounded native chat messages");return 0;
}
static int run_loop(CNGEngine *engine,const CGManifest *m,const CNGInfo *info) {
    char line[512],id[65],extra[2],error[2048];size_t jb=0,bb=0;
    while(fgets(line,sizeof line,stdin)) {
        if(!strcmp(line,"QUIT\n"))return 0;
        if(!strchr(line,'\n')||sscanf(line,"RUN %64s %zu %zu %1s",id,&jb,&bb,extra)!=3||!jb||jb>4096||!bb||bb>m->max_input_bytes+16384||bb>((size_t)64<<20))return 2;
        for(const char *p=id;*p;p++)if(!isalnum((unsigned char)*p)&&*p!='_'&&*p!='-')return 2;
        char *header=malloc(jb+1);unsigned char *body=malloc(bb+1);
        if(!header||!body){free(header);free(body);return 2;}
        if(!read_exact(header,jb)||fgetc(stdin)!='\n'||!read_exact(body,bb)||fgetc(stdin)!='\n'){free(header);free(body);return 2;}
        header[jb]=0;body[bb]=0;jval *options=NULL;
        if(!memchr(header,0,jb)&&!nul_escape(header))options=json_parse_checked(header);free(header);
        const char *operation=jstr(options,"operation");int sent=0,unique=1;
        if(options&&options->t==J_OBJ)for(int a=0;a<options->len;a++)for(int b=0;b<a;b++)if(!strcmp(options->keys[a],options->keys[b]))unique=0;
        if(!options||options->t!=J_OBJ||!operation||!unique)sent=error_frame(id,"invalid native request metadata");
        else if((!strcmp(operation,"complete")||!strcmp(operation,"chat"))&&m->modality==CNG_TEXT) {
            unsigned tokens=0;char *text=NULL,*rendered=NULL;size_t n=0,rendered_bytes=0;
            if(!jint(options,"max_new_tokens",m->max_tokens<128?m->max_tokens:128,m->max_tokens,&tokens))sent=error_frame(id,"invalid native text options");
            else if(!render_prompt(engine,body,bb,!strcmp(operation,"chat"),m->max_input_bytes,&rendered,&rendered_bytes,error,sizeof error))sent=error_frame(id,error);
            else if(cng_text(engine,rendered,rendered_bytes,tokens,&text,&n,error,sizeof error)!=CNG_OK)sent=error_frame(id,error);
            else {size_t prompt=0;cng_last_prompt_tokens(engine,&prompt);sent=text_frame(id,text,n,info,prompt);}cng_free(rendered);cng_free(text);
        } else if(!strcmp(operation,"transcribe")&&m->modality==CNG_AUDIO) {
            nm_wav_info wav;float *samples=NULL;char *text=NULL;size_t n=0;
            const char *task=jstr(options,"task"),*language=jstr(options,"language");jval *lv=json_get(options,"language");
            if(!task||(!(lv&&lv->t==J_NULL)&&(!language||strcmp(language,"en")))||nm_wav_inspect(body,bb,m->max_input_bytes,m->max_audio_seconds,&wav)!=NM_OK)sent=error_frame(id,"invalid native WAV/task/language");
            else if(!(samples=malloc(wav.sample_count*sizeof(float))))sent=error_frame(id,"native audio allocation failed");
            else if(nm_wav_decode(body,bb,m->max_input_bytes,m->max_audio_seconds,samples,wav.sample_count,&wav)!=NM_OK)sent=error_frame(id,"invalid native WAV samples");
            else if(cng_whisper(engine,samples,wav.sample_count,task,language,&text,&n,error,sizeof error)!=CNG_OK)sent=error_frame(id,error);
            else sent=text_frame(id,text,n,info,0);free(samples);cng_free(text);
        } else if(!strcmp(operation,"upscale")&&m->modality==CNG_CAMERA) {
            Bytes png={0};size_t w=0,h=0;
            if(bb>m->max_input_bytes)sent=error_frame(id,"image exceeds model input bound");
            else if(!sr_run(engine,m,info,body,bb,&png,&w,&h,error,sizeof error))sent=error_frame(id,error);
            else {char meta[512];int n=snprintf(meta,sizeof meta,"{\"width\":%zu,\"height\":%zu,\"scale\":4,\"inference_runtime\":\"native-c\",\"device\":\"%s\",\"execution_devices_verified\":%s}",w,h,info->selected_device,info->execution_devices_verified?"true":"false");
                sent=n>0&&(size_t)n<sizeof meta&&frame(id,meta,(size_t)n,png.p,png.n);}free(png.p);
        } else sent=error_frame(id,"request operation does not match this model");
        json_free(options);free(body);if(!sent)return 2;
    }return ferror(stdin)?2:0;
}
static int uint_arg(const char *s,uint64_t *value){if(!s||!*s)return 0;uint64_t n=0;for(;*s;s++){if(*s<'0'||*s>'9'||n>(UINT64_MAX-(unsigned)(*s-'0'))/10)return 0;n=n*10+(unsigned)(*s-'0');}*value=n;return 1;}
static int graph_main(int argc,char **argv,const char *ov,const char *genai,const char *tokenizers) {
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    const char *root=NULL,*device="AUTO",*validate=NULL;
    uint64_t budget=(uint64_t)1<<30,validation_pixels=5592405,validation_bytes=4<<20;
    for(int a=1;a<argc;a++) {
        const char *key=argv[a];if(a+1>=argc){fprintf(stderr,"native_graph: expected value for %s\n",key);return 2;}const char *v=argv[++a];
        if(!strcmp(key,"--model"))root=v;else if(!strcmp(key,"--device"))device=v;
        else if(!strcmp(key,"--ov-c"))ov=v;else if(!strcmp(key,"--genai-c"))genai=v;else if(!strcmp(key,"--tokenizers"))tokenizers=v;
        else if(!strcmp(key,"--memory-budget")){if(!uint_arg(v,&budget))return 2;}
        else if(!strcmp(key,"--validate"))validate=v;else if(!strcmp(key,"--max-pixels")){if(!uint_arg(v,&validation_pixels))return 2;}
        else if(!strcmp(key,"--max-bytes")){if(!uint_arg(v,&validation_bytes))return 2;}
        else {fprintf(stderr,"native_graph: unknown option %s\n",key);return 2;}
    }
    char error[2048]="no eligible native device";
    if(validate) {
        if(strcmp(validate,"camera")||!validation_pixels||validation_pixels>(16u<<20)||!validation_bytes||validation_bytes>(32u<<20))return 2;
        Bytes b={0};b.limit=(size_t)validation_bytes;unsigned char block[65536];size_t n=0;
        while((n=fread(block,1,sizeof block,stdin))!=0)if(!append(&b,block,n))break;
        unsigned char *rgba=NULL;size_t w=0,h=0;int ok=!b.failed&&!ferror(stdin)&&png_decode(b.p,b.n,(size_t)validation_pixels,&rgba,&w,&h,NULL,error,sizeof error);
        stbi_image_free(rgba);free(b.p);if(!ok){fputs("native PNG validation failed\n",stderr);return 2;}
        printf("{\"width\":%zu,\"height\":%zu,\"output_pixels\":%zu}\n",w,h,w*h*16);return 0;
    }
    CGManifest model;if(!root||!cg_manifest_load(root,budget,&model,error,sizeof error)){fprintf(stderr,"native_graph: %s\n",root?error:"--model required");return 2;}
    CNGOptions o;memset(&o,0,sizeof o);o.model_dir=model.model_dir;o.modality=model.modality;o.ov_c_library=ov;o.genai_c_library=genai;o.tokenizers_library=tokenizers;o.sr_xml=model.sr_xml;
    o.max_tokens=model.max_tokens;o.max_prompt_tokens=model.max_prompt_tokens;o.max_context_tokens=model.max_context_tokens;o.inference_f16=model.inference_f16;
    CNGEngine *engine=NULL;
    if(!strcmp(device,"AUTO")) {
        char *available=NULL;size_t count=0;
        if(cng_available_devices(ov,&available,&count,error,sizeof error)!=CNG_OK){fprintf(stderr,"native_graph: %s\n",error);return 2;}
        const char *classes[]={"GPU","NPU","CPU"};
        for(size_t p=0;p<3&&!engine;p++) {
            char *cur=available;
            while(*cur&&!engine){char *end=strchr(cur,'\n');if(end)*end=0;
                if(!strncmp(cur,classes[p],3)&&(cur[3]==0||cur[3]=='.')){o.device=cur;if(cng_create(&o,&engine,error,sizeof error)!=CNG_OK)fprintf(stderr,"native_graph: AUTO %s failed: %s\n",cur,error);}
                if(end){*end='\n';cur=end+1;}else break;}
        }cng_free(available);
    } else {o.device=device;cng_create(&o,&engine,error,sizeof error);}
    if(!engine){fprintf(stderr,"native_graph: %s\n",error);return 2;}
    CNGInfo info;if(cng_info(engine,&info)!=CNG_OK){cng_destroy(engine);return 2;}ready(&model,&info);
    int status=run_loop(engine,&model,&info);cng_destroy(engine);return status;
}
#ifdef _WIN32
static char *wide_utf8(const wchar_t *text) {
    int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,text,-1,NULL,0,NULL,NULL);
    if(n<1||n>16384)return NULL;char *out=malloc((size_t)n);
    if(out&&!WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,text,-1,out,n,NULL,NULL)){free(out);out=NULL;}return out;
}
int main(void) {
    /* The narrow CRT argv uses the active code page. Model paths and all our
     * filesystem/SDK boundaries use UTF-8, so read the original wide arguments. */
    int argc=0,status=2;wchar_t **wide=CommandLineToArgvW(GetCommandLineW(),&argc);
    char **argv=NULL,*libraries[3]={NULL,NULL,NULL};
    if(!wide||argc<1||argc>64)goto done;
    argv=calloc((size_t)argc+1,sizeof *argv);if(!argv)goto done;
    for(int i=0;i<argc;i++)if(!(argv[i]=wide_utf8(wide[i])))goto done;
    const wchar_t *keys[]={L"COLI_OV_C_LIB",L"COLI_GENAI_C_LIB",L"COLI_OV_TOKENIZERS_LIB"};
    for(int i=0;i<3;i++) {
        const wchar_t *value=_wgetenv(keys[i]);
        if(value&&!(libraries[i]=wide_utf8(value)))goto done;
    }
    status=graph_main(argc,argv,libraries[0],libraries[1],libraries[2]);
done:
    if(argv){for(int i=0;i<argc;i++)free(argv[i]);free(argv);}if(wide)LocalFree(wide);
    for(int i=0;i<3;i++)free(libraries[i]);
    if(status==2)fputs("native_graph: check native arguments/dependency diagnostics\n",stderr);
    return status;
}
#else
int main(int argc,char **argv){return graph_main(argc,argv,getenv("COLI_OV_C_LIB"),getenv("COLI_GENAI_C_LIB"),getenv("COLI_OV_TOKENIZERS_LIB"));}
#endif
