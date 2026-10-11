#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#endif
#include "native_graph_manifest.h"
#include "json.h"
#include "evidence_digest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

typedef struct {
    uint64_t size, device, inode, modified;
#ifdef _WIN32
    HANDLE fd;
#else
    int fd;
#endif
} CGFile;

#ifndef _WIN32
static uint64_t modified_time(const struct stat *s) {
#ifdef __APPLE__
    return (uint64_t)s->st_mtimespec.tv_sec*1000000000+(uint64_t)s->st_mtimespec.tv_nsec;
#else
    return (uint64_t)s->st_mtim.tv_sec*1000000000+(uint64_t)s->st_mtim.tv_nsec;
#endif
}
#endif
static int nul_escape(const char *s) {
    for(size_t i=0;s[i];i++)if(s[i]=='\\'&&s[i+1]) {
        if(s[i+1]=='u'&&!strncmp(s+i+2,"0000",4))return 1;i++;
    }return 0;
}

static int fail(char *error,size_t cap,const char *message) {
    if(error&&cap)snprintf(error,cap,"%s",message);
    return 0;
}
static int regular_open(const char *path,int directory,CGFile *f) {
    memset(f,0,sizeof *f);
#ifdef _WIN32
    wchar_t wide[4096];
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide,4096))return 0;
    f->fd=CreateFileW(wide,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_SEQUENTIAL_SCAN,NULL);
    if(f->fd==INVALID_HANDLE_VALUE)return 0;
    BY_HANDLE_FILE_INFORMATION s;
    if(!GetFileInformationByHandle(f->fd,&s)||(s.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)||
       !!(s.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=!!directory){CloseHandle(f->fd);return 0;}
    f->size=((uint64_t)s.nFileSizeHigh<<32)|s.nFileSizeLow;
    f->device=s.dwVolumeSerialNumber;f->inode=((uint64_t)s.nFileIndexHigh<<32)|s.nFileIndexLow;
    f->modified=((uint64_t)s.ftLastWriteTime.dwHighDateTime<<32)|s.ftLastWriteTime.dwLowDateTime;
#else
    struct stat before,s;
    if(lstat(path,&before)||S_ISLNK(before.st_mode)||
       (directory?!S_ISDIR(before.st_mode):!S_ISREG(before.st_mode)))return 0;
    f->fd=open(path,O_RDONLY|O_NONBLOCK
#ifdef O_NOFOLLOW
               |O_NOFOLLOW
#endif
               );
    if(f->fd<0)return 0;
    if(fstat(f->fd,&s)||before.st_dev!=s.st_dev||before.st_ino!=s.st_ino||
       (directory?!S_ISDIR(s.st_mode):!S_ISREG(s.st_mode))){close(f->fd);return 0;}
    f->size=(uint64_t)s.st_size;f->device=(uint64_t)s.st_dev;f->inode=(uint64_t)s.st_ino;
    f->modified=modified_time(&s);
#endif
    return 1;
}
static void file_close(CGFile *f) {
#ifdef _WIN32
    CloseHandle(f->fd);
#else
    close(f->fd);
#endif
}
static int file_read(CGFile *f,void *buffer,size_t bytes,size_t *got) {
#ifdef _WIN32
    DWORD n=0;if(!ReadFile(f->fd,buffer,(DWORD)bytes,&n,NULL))return 0;*got=n;
#else
    ssize_t n=read(f->fd,buffer,bytes);if(n<0)return 0;*got=(size_t)n;
#endif
    return 1;
}
static int unchanged(CGFile *f) {
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION s;
    return GetFileInformationByHandle(f->fd,&s)&&f->size==(((uint64_t)s.nFileSizeHigh<<32)|s.nFileSizeLow)&&
        f->modified==(((uint64_t)s.ftLastWriteTime.dwHighDateTime<<32)|s.ftLastWriteTime.dwLowDateTime);
#else
    struct stat s;return !fstat(f->fd,&s)&&f->size==(uint64_t)s.st_size&&
        f->modified==modified_time(&s);
#endif
}
static int path_join(char *out,size_t cap,const char *root,const char *name) {
    int n=snprintf(out,cap,"%s/%s",root,name);return n>0&&(size_t)n<cap;
}
static int relative_name(const char *name) {
    if(!name||!*name||strlen(name)>1024||name[0]=='/')return 0;
    const char *start=name;
    for(const char *p=name;;p++) {
        unsigned char c=(unsigned char)*p;
        if(c&&c!='/') {if(c<32||c>126||strchr("\\<>:\"|?*",c))return 0;continue;}
        size_t n=(size_t)(p-start);
        if(!n||n>255||(n==1&&*start=='.')||(n==2&&start[0]=='.'&&start[1]=='.')||
           start[n-1]=='.'||start[n-1]==' ')return 0;
        char stem[256];size_t k=0;
        while(k<n&&start[k]!='.'){stem[k]=(char)toupper((unsigned char)start[k]);k++;}stem[k]=0;
        if(!strcmp(stem,"CON")||!strcmp(stem,"PRN")||!strcmp(stem,"AUX")||!strcmp(stem,"NUL")||
           (k==4&&(!memcmp(stem,"COM",3)||!memcmp(stem,"LPT",3))&&stem[3]>='1'&&stem[3]<='9'))return 0;
        if(!c)return 1;start=p+1;
    }
}
static int parents_safe(const char *root,const char *name) {
    char path[4096];if(!path_join(path,sizeof path,root,name))return 0;
    size_t base=strlen(root)+1;
    for(size_t i=base;path[i];i++)if(path[i]=='/') {
        path[i]=0;CGFile d;int ok=regular_open(path,1,&d);path[i]='/';
        if(!ok)return 0;file_close(&d);
    }
    return 1;
}
static int json_valid(jval *v,unsigned depth,size_t *nodes) {
    if(!v||depth>32||++*nodes>65536)return 0;
    if(v->t==J_NUM&&!isfinite(v->num))return 0;
    if(v->t==J_ARR||v->t==J_OBJ) {
        if(v->len<0||v->len>4096)return 0;
        for(int i=0;i<v->len;i++) {
            if(v->t==J_OBJ)for(int j=0;j<i;j++)if(!strcmp(v->keys[i],v->keys[j]))return 0;
            if(!json_valid(v->kids[i],depth+1,nodes))return 0;
        }
    }
    return 1;
}
static const char *string(jval *o,const char *key) {
    jval *v=json_get(o,key);return v&&v->t==J_STR?v->str:NULL;
}
static int number(jval *o,const char *key,uint64_t def,uint64_t low,uint64_t high,uint64_t *out) {
    jval *v=json_get(o,key);if(!v){*out=def;return def>=low&&def<=high;}
    if(v->t!=J_NUM||!isfinite(v->num)||v->num<(double)low||v->num>(double)high||floor(v->num)!=v->num)return 0;
    *out=(uint64_t)v->num;return 1;
}
static int copy_string(char *out,size_t cap,const char *s,int no_spaces) {
    if(!s||!*s||strlen(s)>=cap)return 0;
    for(const char *p=s;*p;p++)if((unsigned char)*p<(no_spaces?33:32)||(unsigned char)*p>126)return 0;
    memcpy(out,s,strlen(s)+1);return 1;
}
static int same_name(const char *a,const char *b) {
    while(*a&&*b&&tolower((unsigned char)*a)==tolower((unsigned char)*b)){a++;b++;}
    return !*a&&!*b;
}
static int listed(jval *files,const char *prefix,const char *name) {
    char combined[2048];if(!path_join(combined,sizeof combined,prefix,name))return 0;
    return json_get(files,*prefix?combined: name)!=NULL;
}
int cg_manifest_load(const char *root,uint64_t budget,CGManifest *out,char *error,size_t cap) {
    if(!root||!*root||strlen(root)>=4096||!out||!budget||budget>((uint64_t)64<<30))return fail(error,cap,"invalid native model directory or budget");
    CGManifest m;memset(&m,0,sizeof m);strcpy(m.root,root);strcpy(m.model_dir,root);
    m.memory_budget_bytes=budget;m.tile_overlap=-1;m.max_audio_seconds=30;
    CGFile directory;if(!regular_open(root,1,&directory))return fail(error,cap,"model directory must be real and not a symlink");file_close(&directory);
    char path[4096];if(!path_join(path,sizeof path,root,"colibri_runtime.json"))return fail(error,cap,"model path too long");
    CGFile f;if(!regular_open(path,0,&f))return fail(error,cap,"native model manifest is missing or not regular");
    if(!f.size||f.size>65536){file_close(&f);return fail(error,cap,"native manifest exceeds its byte bound");}
    size_t bytes=(size_t)f.size,got=0,total=0;char *raw=malloc(bytes+1);
    if(!raw){file_close(&f);return fail(error,cap,"manifest allocation failed");}
    while(total<bytes&&file_read(&f,raw+total,bytes-total,&got)&&got)total+=got;
    int same=unchanged(&f);file_close(&f);raw[bytes]=0;
    if(total!=bytes||!same||memchr(raw,0,bytes)||nul_escape(raw)){free(raw);return fail(error,cap,"invalid or changed manifest bytes");}
    jval *j=json_parse_checked(raw);free(raw);size_t nodes=0;
    if(!j||j->t!=J_OBJ||!json_valid(j,0,&nodes)){json_free(j);return fail(error,cap,"invalid, duplicate or nonfinite manifest JSON");}
    const char *schema=string(j,"schema"),*family=string(j,"family"),*adapter=string(j,"adapter");
    int ok=schema&&!strcmp(schema,"colibri.runtime-model/v1")&&family&&adapter;
    if(ok&&!strcmp(family,"text_graph")&&!strcmp(adapter,"text_openvino"))m.modality=CNG_TEXT;
    else if(ok&&!strcmp(family,"speech")&&!strcmp(adapter,"whisper_openvino"))m.modality=CNG_AUDIO;
    else if(ok&&!strcmp(family,"camera_upscale")&&!strcmp(adapter,"super_resolution_openvino"))m.modality=CNG_CAMERA;
    else ok=0;
    ok=ok&&copy_string(m.family,sizeof m.family,family,1)&&copy_string(m.model_id,sizeof m.model_id,string(j,"model_id"),1)&&
       copy_string(m.precision,sizeof m.precision,string(j,"precision"),0);
    jval *files=json_get(j,"files"),*options=json_get(j,"options");
    ok=ok&&files&&files->t==J_OBJ&&files->len>=1&&files->len<=1024&&options&&options->t==J_OBJ;
    uint64_t input=0,resident=0,workspace=0,session=0,value=0;
    const char *required[]={"max_input_bytes","resident_bytes","workspace_bytes","session_bytes"};
    for(size_t i=0;i<4;i++)if(!json_get(j,required[i]))ok=0;
    ok=ok&&number(j,"max_input_bytes",0,1,64<<20,&input)&&number(j,"resident_bytes",0,0,budget,&resident)&&
       number(j,"workspace_bytes",0,0,budget,&workspace)&&number(j,"session_bytes",0,0,budget,&session);
    (void)session;m.max_input_bytes=(size_t)input;
    m.max_output_bytes=32<<20;m.max_output_pixels=5592405;m.max_tokens=m.modality==CNG_AUDIO?448:256;
    m.max_prompt_tokens=2048;m.max_context_tokens=4096;
    if(ok) {
        ok=number(options,m.modality==CNG_AUDIO?"max_new_tokens":"max_tokens",m.max_tokens,1,m.max_tokens,&value);m.max_tokens=(unsigned)value;
        ok=ok&&number(options,"max_prompt_tokens",2048,1,4096,&value);m.max_prompt_tokens=(unsigned)value;
        ok=ok&&number(options,"max_context_tokens",4096,2,4096,&value);m.max_context_tokens=(unsigned)value;
        ok=ok&&number(options,"max_output_bytes",32<<20,1024,128<<20,&value);m.max_output_bytes=(size_t)value;
        ok=ok&&number(options,"max_output_pixels",5592405,1,16<<20,&value);m.max_output_pixels=(size_t)value;
        if(json_get(options,"tile_overlap")){ok=ok&&number(options,"tile_overlap",0,0,511,&value);m.tile_overlap=(int)value;}
        if(json_get(options,"scale")){ok=ok&&number(options,"scale",4,4,4,&value);}
        const char *hint=string(options,"inference_precision");
        if(json_get(options,"inference_precision")&&!hint)ok=0;
        if(hint){if(strcmp(hint,"f16")&&strcmp(hint,"f32"))ok=0;m.inference_f16=!strcmp(hint,"f16");}
        jval *duration=json_get(options,"max_audio_seconds");
        if(duration){ok=ok&&duration->t==J_NUM&&isfinite(duration->num)&&duration->num>0&&duration->num<=30;m.max_audio_seconds=duration->num;}
    }
    if(!ok){json_free(j);return fail(error,cap,"native manifest fields/options are invalid");}
    uint64_t seen_dev[1024],seen_ino[1024];
    for(int i=0;i<files->len;i++) {
        const char *name=files->keys[i];jval *digest=files->kids[i];
        if(!relative_name(name)||same_name(name,"colibri_runtime.json")||!digest||digest->t!=J_STR||strlen(digest->str)!=64){ok=0;break;}
        for(size_t k=0;k<64;k++)if(!strchr("0123456789abcdef",digest->str[k]))ok=0;
        for(int k=0;k<i;k++) {
            const char *a=name,*b=files->keys[k];
            while(*a&&*b&&tolower((unsigned char)*a)==tolower((unsigned char)*b)){a++;b++;}
            if(!*a&&!*b)ok=0;
        }
        if(!ok||!parents_safe(root,name)||!path_join(path,sizeof path,root,name)||!regular_open(path,0,&f)){ok=0;break;}
        if(!f.size||f.size>budget-m.asset_bytes){file_close(&f);ok=0;break;}
        for(int k=0;k<i;k++)if(f.inode&&f.inode==seen_ino[k]&&f.device==seen_dev[k])ok=0;
        seen_dev[i]=f.device;seen_ino[i]=f.inode;
        EvidenceSha256 sha;unsigned char block[65536],raw_digest[32];evidence_sha256_init(&sha);uint64_t consumed=0;
        while(ok&&file_read(&f,block,sizeof block,&got)&&got){consumed+=got;if(consumed>f.size){ok=0;break;}evidence_sha256_update(&sha,block,got);}
        ok=ok&&consumed==f.size&&unchanged(&f);m.asset_bytes+=f.size;file_close(&f);
        evidence_sha256_final(&sha,raw_digest);char hex[65];
        for(size_t k=0;k<32;k++){hex[k*2]="0123456789abcdef"[raw_digest[k]>>4];hex[k*2+1]="0123456789abcdef"[raw_digest[k]&15];}hex[64]=0;
        if(strcmp(hex,digest->str))ok=0;
        if(!ok)break;
    }
    uint64_t held=m.asset_bytes>resident?m.asset_bytes:resident;
    if(!ok||held>budget||workspace>budget-held){json_free(j);return fail(error,cap,"native model asset identity/path/admission check failed");}
    m.base_estimated_bytes=held+workspace;
    const char *prefix=string(options,"model_path");
    if(json_get(options,"model_path")&&!prefix){json_free(j);return fail(error,cap,"model_path must be a string");}
    if(!prefix||!strcmp(prefix,"."))prefix="";
    if(*prefix&&(!relative_name(prefix)||!parents_safe(root,prefix)||!path_join(m.model_dir,sizeof m.model_dir,root,prefix))){json_free(j);return fail(error,cap,"invalid model_path");}
    if(m.modality==CNG_CAMERA) {
        const char *xml=string(options,"model_xml");
        if(!xml||!json_get(files,xml)||!path_join(m.sr_xml,sizeof m.sr_xml,root,xml)){json_free(j);return fail(error,cap,"camera model_xml must be a declared trained asset");}
        size_t n=strlen(xml);char bin[1025];
        if(n<4||strcmp(xml+n-4,".xml")||n>=sizeof bin){json_free(j);return fail(error,cap,"camera model_xml must end in .xml");}
        strcpy(bin,xml);strcpy(bin+n-4,".bin");if(!json_get(files,bin)){json_free(j);return fail(error,cap,"camera weights must be declared");}
    } else {
        const char *text_files[]={"config.json","generation_config.json","openvino_model.xml","openvino_model.bin","openvino_tokenizer.xml","openvino_tokenizer.bin","openvino_detokenizer.xml","openvino_detokenizer.bin"};
        const char *audio_files[]={"config.json","generation_config.json","preprocessor_config.json","openvino_encoder_model.xml","openvino_encoder_model.bin","openvino_decoder_model.xml","openvino_decoder_model.bin","openvino_tokenizer.xml","openvino_tokenizer.bin","openvino_detokenizer.xml","openvino_detokenizer.bin"};
        const char **required_files=m.modality==CNG_TEXT?text_files:audio_files;size_t n=m.modality==CNG_TEXT?8:11;
        for(size_t i=0;i<n;i++)if(!listed(files,prefix,required_files[i])){json_free(j);return fail(error,cap,"native pipeline is missing a declared required asset");}
    }
    json_free(j);*out=m;if(error&&cap)*error=0;return 1;
}
