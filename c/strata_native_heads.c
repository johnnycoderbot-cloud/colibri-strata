#define _POSIX_C_SOURCE 200809L
#include "strata_native_heads.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "compile strata_native_heads.c separately without fast-math"
#endif
#include "strata_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#if !defined(SN_FORCE_SCALAR) && (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SN_HAVE_AVX2 1
#endif

typedef struct { uint32_t h[8]; uint64_t bytes; unsigned used; uint8_t block[64]; } sn_sha;
static uint32_t sn_rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static const uint32_t sn_k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static void sn_sha_block(sn_sha *s) {
    uint32_t w[64], a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint32_t e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];
    unsigned i;
    for(i=0;i<16;i++) w[i]=(uint32_t)s->block[i*4]<<24|(uint32_t)s->block[i*4+1]<<16|
                            (uint32_t)s->block[i*4+2]<<8|s->block[i*4+3];
    for(i=16;i<64;i++) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=w[i-16]+(sn_rotr(x,7)^sn_rotr(x,18)^(x>>3))+w[i-7]+(sn_rotr(y,17)^sn_rotr(y,19)^(y>>10));
    }
    for(i=0;i<64;i++) {
        uint32_t t1=h+(sn_rotr(e,6)^sn_rotr(e,11)^sn_rotr(e,25))+((e&f)^(~e&g))+sn_k[i]+w[i];
        uint32_t t2=(sn_rotr(a,2)^sn_rotr(a,13)^sn_rotr(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sn_sha_init(sn_sha *s) {
    const uint32_t initial[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h,initial,sizeof(initial));s->bytes=0;s->used=0;
}
static void sn_sha_update(sn_sha *s,const void *data,size_t n) {
    const uint8_t *p=data;s->bytes+=(uint64_t)n;
    while(n) { size_t take=64-s->used;if(take>n)take=n;
        memcpy(s->block+s->used,p,take);s->used+=(unsigned)take;p+=take;n-=take;
        if(s->used==64){sn_sha_block(s);s->used=0;}
    }
}
static void sn_sha_final(sn_sha *s,uint8_t out[32]) {
    uint64_t bits=s->bytes*8;unsigned i;
    s->block[s->used++]=0x80;
    if(s->used>56){memset(s->block+s->used,0,64-s->used);sn_sha_block(s);s->used=0;}
    memset(s->block+s->used,0,56-s->used);
    for(i=0;i<8;i++)s->block[63-i]=(uint8_t)(bits>>(i*8));
    sn_sha_block(s);
    for(i=0;i<8;i++){out[i*4]=(uint8_t)(s->h[i]>>24);out[i*4+1]=(uint8_t)(s->h[i]>>16);
        out[i*4+2]=(uint8_t)(s->h[i]>>8);out[i*4+3]=(uint8_t)s->h[i];}
}
static uint32_t sn_u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t sn_u64(const uint8_t *p){return (uint64_t)sn_u32(p)|(uint64_t)sn_u32(p+4)<<32;}
static void sn_put64(uint8_t p[8],uint64_t v){unsigned i;for(i=0;i<8;i++)p[i]=(uint8_t)(v>>(i*8));}
static void sn_sha64(sn_sha *s,uint64_t v){uint8_t p[8];sn_put64(p,v);sn_sha_update(s,p,8);}

#ifdef _WIN32
static wchar_t *sn_wide(const char *path) {
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);wchar_t *p;
    if(!n)return NULL;
    p=malloc((size_t)n*sizeof(*p));if(!p)return NULL;
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,p,n)){free(p);return NULL;}return p;
}
static FILE *sn_file(const char *path){
    wchar_t *w=sn_wide(path);DWORD attributes;HANDLE handle;int descriptor;FILE *f;
    if(!w)return NULL;
    attributes=GetFileAttributesW(w);
    if(attributes==INVALID_FILE_ATTRIBUTES||(attributes&FILE_ATTRIBUTE_DIRECTORY)){free(w);return NULL;}
    handle=CreateFileW(w,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    free(w);
    if(handle==INVALID_HANDLE_VALUE)return NULL;
    if(GetFileType(handle)!=FILE_TYPE_DISK){CloseHandle(handle);return NULL;}
    descriptor=_open_osfhandle((intptr_t)handle,_O_RDONLY|_O_BINARY);
    if(descriptor<0){CloseHandle(handle);return NULL;}
    f=_fdopen(descriptor,"rb");if(!f)_close(descriptor);return f;
}
#else
static FILE *sn_file(const char *path){
    int flags=O_RDONLY|O_NONBLOCK,descriptor;struct stat st;FILE *f;
#ifdef O_CLOEXEC
    flags|=O_CLOEXEC;
#endif
    descriptor=open(path,flags);if(descriptor<0)return NULL;
    if(fstat(descriptor,&st)||!S_ISREG(st.st_mode)){close(descriptor);return NULL;}
    f=fdopen(descriptor,"rb");if(!f)close(descriptor);return f;
}
#endif
static int sn_file_size(FILE *f,uint64_t *size) {
#ifdef _WIN32
    __int64 n;if(_fseeki64(f,0,SEEK_END))return 0;n=_ftelli64(f);
    if(n<0||_fseeki64(f,0,SEEK_SET))return 0;
    *size=(uint64_t)n;
#else
    off_t n;if(fseeko(f,0,SEEK_END))return 0;n=ftello(f);
    if(n<0||fseeko(f,0,SEEK_SET))return 0;
    *size=(uint64_t)n;
#endif
    return 1;
}
static char *sn_join(const char *root,const char *name){size_t a=strlen(root),b=strlen(name);char *p;
    if(a>SIZE_MAX-b-2)return NULL;
    p=malloc(a+b+2);if(!p)return NULL;
    memcpy(p,root,a);p[a]='/';memcpy(p+a+1,name,b+1);return p;
}
static int sn_selected(const char *name){size_t n=strlen(name);return !strcmp(name,"config.json")||!strcmp(name,"tokenizer.json")||
    !strcmp(name,"qwen36_meta.json")||(n>12&&!strcmp(name+n-12,".safetensors"));}
static int sn_name_compare(const void *a,const void *b){return strcmp(*(const char *const *)a,*(const char *const *)b);}
static sn_status sn_add_name(char ***names,size_t *count,const char *name){size_t n=strlen(name);char *copy;char **next;
    if(*count>=4096||!n||n>4096)return SN_LIMIT;
    copy=malloc(n+1);if(!copy)return SN_NOMEM;memcpy(copy,name,n+1);
    next=realloc(*names,(*count+1)*sizeof(*next));if(!next){free(copy);return SN_NOMEM;}
    *names=next;next[(*count)++]=copy;return SN_OK;
}
sn_status sn_fingerprint_model(const char *root,const char *precision,uint8_t out[32]) {
    char **names=NULL;size_t count=0,i,plen=0;sn_status status=SN_OK;sn_sha hash;uint8_t digest[32];
    int config=0,tokenizer=0,weights=0;
    if(!root||!root[0]||!precision||!out)return SN_INVALID;
    while(plen<=1024&&precision[plen])plen++;
    if(!plen||plen>1024)return SN_INVALID;
#ifdef _WIN32
    { char *pattern=sn_join(root,"*");wchar_t *wide;WIN32_FIND_DATAW data;HANDLE handle;
      if(!pattern)return SN_NOMEM;
      wide=sn_wide(pattern);free(pattern);if(!wide)return SN_IO;
      handle=FindFirstFileW(wide,&data);free(wide);if(handle==INVALID_HANDLE_VALUE)return SN_IO;
      do { if(!(data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)) {
          int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,data.cFileName,-1,NULL,0,NULL,NULL);char *name;
          if(n<=0){status=SN_IO;break;}name=malloc((size_t)n);if(!name){status=SN_NOMEM;break;}
          WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,data.cFileName,-1,name,n,NULL,NULL);
          if(sn_selected(name))status=sn_add_name(&names,&count,name);
          free(name);if(status!=SN_OK)break;
      }}while(FindNextFileW(handle,&data));FindClose(handle);
    }
#else
    { DIR *directory=opendir(root);struct dirent *entry;if(!directory)return SN_IO;
      while((entry=readdir(directory))!=NULL){if(sn_selected(entry->d_name)){status=sn_add_name(&names,&count,entry->d_name);if(status!=SN_OK)break;}}
      closedir(directory);
    }
#endif
    if(status!=SN_OK)goto cleanup;
    qsort(names,count,sizeof(*names),sn_name_compare);
    for(i=0;i<count;i++){config|=!strcmp(names[i],"config.json");tokenizer|=!strcmp(names[i],"tokenizer.json");
        if(strlen(names[i])>12&&!strcmp(names[i]+strlen(names[i])-12,".safetensors"))weights++;}
    if(!config||!tokenizer||!weights){status=SN_INVALID;goto cleanup;}
    sn_sha_init(&hash);sn_sha_update(&hash,"CSNH-model-v1\0",14);sn_sha64(&hash,plen);sn_sha_update(&hash,precision,plen);sn_sha64(&hash,count);
    for(i=0;i<count;i++){
        char *path=sn_join(root,names[i]);FILE *f;uint64_t size,left;uint8_t buffer[65536];
        if(!path){status=SN_NOMEM;goto cleanup;}f=sn_file(path);free(path);if(!f){status=SN_IO;goto cleanup;}
        if(!sn_file_size(f,&size)){fclose(f);status=SN_IO;goto cleanup;}
        if(size>UINT64_MAX/8-hash.bytes){fclose(f);status=SN_LIMIT;goto cleanup;}
        sn_sha64(&hash,strlen(names[i]));sn_sha_update(&hash,names[i],strlen(names[i]));sn_sha64(&hash,size);
        left=size;while(left){size_t chunk=left>sizeof(buffer)?sizeof(buffer):(size_t)left;
            if(fread(buffer,1,chunk,f)!=chunk){status=SN_IO;break;}sn_sha_update(&hash,buffer,chunk);left-=chunk;}
        if(status==SN_OK&&(fgetc(f)!=EOF||ferror(f)))status=SN_IO;
        fclose(f);if(status!=SN_OK)goto cleanup;
    }
    sn_sha_final(&hash,digest);memcpy(out,digest,32);
cleanup:
    for(i=0;i<count;i++)free(names[i]);
    free(names);return status;
}

typedef double (*sn_dot)(const float *,const float *,size_t);
typedef struct {size_t rank,latent_start;float *down,*up,*bias;} sn_head;
struct sn_bank {size_t hidden,vocab,count,total_rank,decoded;unsigned bits;sn_dot dot;const char *kernel;sn_head heads[SN_MAX_HEADS];};
static double sn_dot_scalar(const float *a,const float *b,size_t n){size_t i;double sum=0;for(i=0;i<n;i++)sum+=(double)a[i]*b[i];return sum;}
#ifdef SN_HAVE_AVX2
__attribute__((target("avx2"))) static double sn_dot_avx2(const float *a,const float *b,size_t n){
    size_t i=0;__m256d low=_mm256_setzero_pd(),high=_mm256_setzero_pd();double lanes[4],sum;
    for(;i+8<=n;i+=8){__m256 aa=_mm256_loadu_ps(a+i),bb=_mm256_loadu_ps(b+i);
        low=_mm256_add_pd(low,_mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(aa)),_mm256_cvtps_pd(_mm256_castps256_ps128(bb))));
        high=_mm256_add_pd(high,_mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(aa,1)),_mm256_cvtps_pd(_mm256_extractf128_ps(bb,1))));}
    _mm256_storeu_pd(lanes,_mm256_add_pd(low,high));sum=lanes[0]+lanes[1]+lanes[2]+lanes[3];
    for(;i<n;i++)sum+=(double)a[i]*b[i];
    return sum;
}
#endif
void sn_close(sn_bank *bank){size_t i;if(!bank)return;for(i=0;i<bank->count;i++){free(bank->heads[i].down);free(bank->heads[i].up);free(bank->heads[i].bias);}free(bank);}
static sn_status sn_quant_status(coli_strata_status s){if(s==COLI_STRATA_OK)return SN_OK;if(s==COLI_STRATA_NONFINITE)return SN_NONFINITE;
    if(s==COLI_STRATA_OVERFLOW)return SN_OVERFLOW;
    if(s==COLI_STRATA_CAPACITY)return SN_LIMIT;
    return SN_INVALID;}
sn_status sn_open(const char *path,const uint8_t expected[32],size_t hidden,size_t vocab,sn_bank **out){
    FILE *file;uint64_t length;uint8_t *bytes=NULL,digest[32];sn_sha hash;sn_bank *bank=NULL;sn_status status=SN_INVALID;
    size_t cursor=SN_HEADER_BYTES,i;
    if(!path||!expected||!out||*out||!hidden||!vocab||hidden>INT_MAX||vocab>INT_MAX)return SN_INVALID;
    file=sn_file(path);if(!file)return SN_IO;
    if(!sn_file_size(file,&length)){fclose(file);return SN_IO;}
    if(length<SN_HEADER_BYTES||length>SN_MAX_FILE_BYTES){fclose(file);return SN_LIMIT;}
    bytes=malloc((size_t)length);if(!bytes){fclose(file);return SN_NOMEM;}
    if(fread(bytes,1,(size_t)length,file)!=(size_t)length||fgetc(file)!=EOF||ferror(file)){fclose(file);free(bytes);return SN_IO;}fclose(file);
    if(memcmp(bytes,"CSNH",4)||sn_u32(bytes+4)!=1||sn_u32(bytes+8)!=SN_HEADER_BYTES||sn_u32(bytes+28))goto done;
    if(memcmp(bytes+32,expected,32)){status=SN_IDENTITY;goto done;}
    if(sn_u32(bytes+12)!=hidden||sn_u32(bytes+16)!=vocab){status=SN_DIMENSIONS;goto done;}
    if(!sn_u32(bytes+20)||sn_u32(bytes+20)>SN_MAX_HEADS){status=SN_LIMIT;goto done;}
    if(sn_u32(bytes+24)!=4&&sn_u32(bytes+24)!=8&&sn_u32(bytes+24)!=16)goto done;
    sn_sha_init(&hash);sn_sha_update(&hash,bytes+SN_HEADER_BYTES,(size_t)length-SN_HEADER_BYTES);sn_sha_final(&hash,digest);
    if(memcmp(bytes+64,digest,32))goto done;
    bank=calloc(1,sizeof(*bank));if(!bank){status=SN_NOMEM;goto done;}
    bank->hidden=hidden;bank->vocab=vocab;bank->count=sn_u32(bytes+20);bank->bits=sn_u32(bytes+24);
    bank->dot=sn_dot_scalar;bank->kernel="scalar";
#ifdef SN_HAVE_AVX2
    __builtin_cpu_init();if(__builtin_cpu_supports("avx2")){bank->dot=sn_dot_avx2;bank->kernel="avx2";}
#endif
    for(i=0;i<bank->count;i++){
        sn_head *head=&bank->heads[i];uint64_t down_bytes,up_bytes;size_t down_count,up_count,total_values,decoded,j;
        strata_quant_info info;const uint8_t *down,*up,*bias;
        if((size_t)length-cursor<32)goto done;
        head->rank=sn_u32(bytes+cursor+4);down_bytes=sn_u64(bytes+cursor+8);up_bytes=sn_u64(bytes+cursor+16);
        if(sn_u32(bytes+cursor)!=i+2||!head->rank||head->rank>SN_MAX_RANK||sn_u32(bytes+cursor+24)!=vocab||sn_u32(bytes+cursor+28))goto done;
        cursor+=32;
        if(head->rank>SIZE_MAX/hidden||head->rank>SIZE_MAX/vocab){status=SN_OVERFLOW;goto done;}
        down_count=head->rank*hidden;up_count=head->rank*vocab;
        if(down_count>STRATA_QUANT_MAX_VALUES||up_count>STRATA_QUANT_MAX_VALUES){status=SN_LIMIT;goto done;}
        if(down_bytes>(size_t)length-cursor)goto done;
        down=bytes+cursor;cursor+=(size_t)down_bytes;
        if(up_bytes>(size_t)length-cursor)goto done;
        up=bytes+cursor;cursor+=(size_t)up_bytes;
        if(vocab>((size_t)length-cursor)/4)goto done;
        bias=bytes+cursor;cursor+=vocab*4;
        status=sn_quant_status(strata_quant_inspect(down,(size_t)down_bytes,&info));if(status!=SN_OK)goto done;
        if(info.count!=down_count||info.bits!=bank->bits){status=SN_INVALID;goto done;}
        status=sn_quant_status(strata_quant_inspect(up,(size_t)up_bytes,&info));if(status!=SN_OK)goto done;
        if(info.count!=up_count||info.bits!=bank->bits){status=SN_INVALID;goto done;}
        if(down_count>SIZE_MAX-up_count||down_count+up_count>SIZE_MAX-vocab){status=SN_OVERFLOW;goto done;}
        total_values=down_count+up_count+vocab;
        if(total_values>(SN_MAX_DECODED_BYTES-bank->decoded)/sizeof(float)){status=SN_LIMIT;goto done;}
        head->down=malloc(down_count*sizeof(float));head->up=malloc(up_count*sizeof(float));head->bias=malloc(vocab*sizeof(float));
        if(!head->down||!head->up||!head->bias){status=SN_NOMEM;goto done;}
        status=sn_quant_status(strata_quant_decode(down,(size_t)down_bytes,head->down,down_count,&decoded));if(status!=SN_OK)goto done;
        status=sn_quant_status(strata_quant_decode(up,(size_t)up_bytes,head->up,up_count,&decoded));if(status!=SN_OK)goto done;
        for(j=0;j<vocab;j++){uint32_t bits=sn_u32(bias+j*4);memcpy(&head->bias[j],&bits,4);
            if(!isfinite(head->bias[j])){status=SN_NONFINITE;goto done;}}
        head->latent_start=bank->total_rank;bank->total_rank+=head->rank;bank->decoded+=total_values*sizeof(float);
        status=SN_INVALID; /* Parsing failures in the next record must stay failures. */
    }
    if(cursor!=(size_t)length)goto done;
    *out=bank;bank=NULL;status=SN_OK;
done:sn_close(bank);free(bytes);return status;
}
sn_status sn_get_info(const sn_bank *bank,sn_info *out){sn_info info;if(!bank||!out)return SN_INVALID;
    info.hidden_dim=bank->hidden;info.vocab_size=bank->vocab;info.head_count=bank->count;info.decoded_bytes=bank->decoded;
    info.source_bits=bank->bits;info.kernel=bank->kernel;*out=info;return SN_OK;}
sn_status sn_predict_sized(const sn_bank *bank,const float *hidden,size_t n,int *out,size_t capacity,int parallel){
    float latent[SN_MAX_HEADS*SN_MAX_RANK],best[SN_MAX_HEADS];int tokens[SN_MAX_HEADS],seen[SN_MAX_HEADS]={0};
    size_t i;int error=0;
    if(!bank||!hidden||!out)return SN_INVALID;
    if(n!=bank->hidden||capacity<bank->count)return SN_DIMENSIONS;
    if(n>SIZE_MAX/sizeof(float))return SN_OVERFLOW;
    if(coli_strata_overlap_(hidden,n*sizeof(float),out,bank->count*sizeof(int)))return SN_INVALID;
    for(i=0;i<n;i++)if(!isfinite(hidden[i]))return SN_NONFINITE;
#ifndef _OPENMP
    (void)parallel;
#else
#pragma omp parallel if(parallel && !omp_in_parallel())
#endif
    {
        float local_best[SN_MAX_HEADS];int local_token[SN_MAX_HEADS],local_seen[SN_MAX_HEADS]={0},local_error=0;
        size_t task,h;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(task=0;task<bank->total_rank;task++){
            double value;h=0;while(h+1<bank->count&&task>=bank->heads[h+1].latent_start)h++;
            value=bank->dot(bank->heads[h].down+(task-bank->heads[h].latent_start)*n,hidden,n);
            if(!isfinite(value)||value>FLT_MAX||value<-FLT_MAX){
#ifdef _OPENMP
#pragma omp atomic write
#endif
                error=1;
            }else latent[task]=(float)value;
        }
        if(!error){
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for(task=0;task<bank->count*bank->vocab;task++){
                size_t v=task%bank->vocab;double value;float score;h=task/bank->vocab;
                value=bank->dot(bank->heads[h].up+v*bank->heads[h].rank,latent+bank->heads[h].latent_start,bank->heads[h].rank)+(double)bank->heads[h].bias[v];
                if(!isfinite(value)||value>FLT_MAX||value<-FLT_MAX){local_error=1;continue;}
                score=(float)value;
                if(!local_seen[h]||score>local_best[h]||(score==local_best[h]&&(int)v<local_token[h])){
                    local_seen[h]=1;local_best[h]=score;local_token[h]=(int)v;}
            }
        }
#ifdef _OPENMP
#pragma omp critical(sn_heads_publish)
#endif
        { if(local_error)error=1;
          for(h=0;h<bank->count;h++)if(local_seen[h]&&(!seen[h]||local_best[h]>best[h]||(local_best[h]==best[h]&&local_token[h]<tokens[h]))){
              seen[h]=1;best[h]=local_best[h];tokens[h]=local_token[h];}
        }
    }
    if(error)return SN_OVERFLOW;
    for(i=0;i<bank->count;i++)if(!seen[i])return SN_INVALID;
    memcpy(out,tokens,bank->count*sizeof(*out));return SN_OK;
}
sn_status sn_predict(const sn_bank *bank,const float *hidden,int *out){return bank?sn_predict_sized(bank,hidden,bank->hidden,out,bank->count,1):SN_INVALID;}
const char *sn_status_string(sn_status s){switch(s){case SN_OK:return "ok";case SN_INVALID:return "invalid head pack or arguments";
    case SN_IO:return "head pack/model I/O failure";case SN_IDENTITY:return "head pack belongs to another model/precision";
    case SN_DIMENSIONS:return "head dimensions/capacity mismatch";case SN_LIMIT:return "head pack exceeds bounds";
    case SN_NONFINITE:return "nonfinite head input/weights";case SN_OVERFLOW:return "head arithmetic/size overflow";
    case SN_NOMEM:return "head allocation failure";default:return "unknown head failure";}}
