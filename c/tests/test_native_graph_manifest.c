/* SDK-free, real filesystem manifest gates. Assets are deliberately small
 * structural fixtures; no neural model or device is loaded. SHA constants
 * were independently computed with Python hashlib over b"xml" / b"bin". */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#endif
#include "../native_graph_manifest.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>
#ifndef SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
#define SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE 2
#endif
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char sha_xml[]="bf92a04e60b4b2362d45490e5142e47b687c2ebb5898da6bc71b697602016e6e";
static const char sha_bin[]="51a1f05af85e342e3c849b47d387086476282d5f50dc240c19216d6edfb1eb5a";
static char root[2048];
static unsigned failures,checks;
static void expect(int condition,const char *label){
    checks++;if(!condition){fprintf(stderr,"manifest gate failed: %s\n",label);failures++;}
}
static void join(char *out,size_t cap,const char *base,const char *name){
    int n=snprintf(out,cap,"%s/%s",base,name);assert(n>0&&(size_t)n<cap);
}
#ifdef _WIN32
static void wide_path(const char *path,wchar_t out[4096]){
    assert(MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,out,4096));
}
#endif
static FILE *open_file(const char *name,const char *mode){
    char path[4096];join(path,sizeof path,root,name);
#ifdef _WIN32
    wchar_t wp[4096],wm[16];wide_path(path,wp);
    assert(MultiByteToWideChar(CP_UTF8,0,mode,-1,wm,16));return _wfopen(wp,wm);
#else
    return fopen(path,mode);
#endif
}
static void write_file(const char *name,const void *bytes,size_t count){
    FILE *f=open_file(name,"wb");assert(f);assert(fwrite(bytes,1,count,f)==count);assert(!fclose(f));
}
static void unlink_file(const char *name){
    char path[4096];join(path,sizeof path,root,name);
#ifdef _WIN32
    wchar_t wp[4096];wide_path(path,wp);assert(DeleteFileW(wp));
#else
    assert(!unlink(path));
#endif
}
static void mkdir_name(const char *name){
    char path[4096];join(path,sizeof path,root,name);
#ifdef _WIN32
    wchar_t wp[4096];wide_path(path,wp);assert(CreateDirectoryW(wp,NULL));
#else
    assert(!mkdir(path,0700));
#endif
}
static void rmdir_name(const char *name){
    char path[4096];join(path,sizeof path,root,name);
#ifdef _WIN32
    wchar_t wp[4096];wide_path(path,wp);assert(RemoveDirectoryW(wp));
#else
    assert(!rmdir(path));
#endif
}
static int symlink_name(const char *name,const char *target,int directory){
    char link_path[4096],target_path[4096];join(link_path,sizeof link_path,root,name);join(target_path,sizeof target_path,root,target);
#ifdef _WIN32
    wchar_t wl[4096],wt[4096];wide_path(link_path,wl);wide_path(target_path,wt);
    DWORD flags=(directory?SYMBOLIC_LINK_FLAG_DIRECTORY:0)|SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
    if(CreateSymbolicLinkW(wl,wt,flags))return 1;
    if(GetLastError()==ERROR_INVALID_PARAMETER&&CreateSymbolicLinkW(wl,wt,flags&~SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE))return 1;
    fprintf(stderr,"symlink gate unavailable on this Windows account (%lu)\n",(unsigned long)GetLastError());return 0;
#else
    (void)directory;return symlink(target_path,link_path)==0;
#endif
}
static void hardlink_name(const char *name,const char *target){
    char link_path[4096],target_path[4096];join(link_path,sizeof link_path,root,name);join(target_path,sizeof target_path,root,target);
#ifdef _WIN32
    wchar_t wl[4096],wt[4096];wide_path(link_path,wl);wide_path(target_path,wt);assert(CreateHardLinkW(wl,wt,NULL));
#else
    assert(!link(target_path,link_path));
#endif
}
static void setup(void){
#ifdef _WIN32
    wchar_t temp[2048],path[2048];assert(GetTempPathW(2048,temp));assert(GetTempFileNameW(temp,L"cgm",0,path));
    assert(DeleteFileW(path));assert(CreateDirectoryW(path,NULL));
    assert(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,path,-1,root,sizeof root,NULL,NULL));
#else
    strcpy(root,"/tmp/colibri-native-manifest-XXXXXX");assert(mkdtemp(root));
#endif
    write_file("model.xml","xml",3);write_file("model.bin","bin",3);
}
static void cleanup(void){
    unlink_file("colibri_runtime.json");unlink_file("model.xml");unlink_file("model.bin");
#ifdef _WIN32
    wchar_t wp[4096];wide_path(root,wp);assert(RemoveDirectoryW(wp));
#else
    assert(!rmdir(root));
#endif
}
static void camera_files(char *out,size_t capacity){
    int n=snprintf(out,capacity,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\"}",sha_xml,sha_bin);
    assert(n>0&&(size_t)n<capacity);
}
static void manifest(const char *family,const char *adapter,const char *files,const char *options,const char *extra){
    char document[16384];
    int n=snprintf(document,sizeof document,
        "{\"schema\":\"colibri.runtime-model/v1\",\"family\":\"%s\",\"adapter\":\"%s\","
        "\"model_id\":\"camera-fixture\",\"precision\":\"f32\",\"max_input_bytes\":4096,"
        "\"resident_bytes\":128,\"workspace_bytes\":256,\"session_bytes\":64,\"files\":%s,\"options\":%s%s}",
        family,adapter,files,options,extra?extra:"");
    assert(n>0&&(size_t)n<sizeof document);write_file("colibri_runtime.json",document,(size_t)n);
}
static void camera(const char *files,const char *options,const char *extra){
    manifest("camera_upscale","super_resolution_openvino",files,options,extra);
}
static void rejects_at(const char *directory,uint64_t budget,const char *label){
    struct {uint64_t before;CGManifest value;uint64_t after;} box;
    memset(&box,0x79,sizeof box);unsigned char saved[sizeof box];memcpy(saved,&box,sizeof box);
    char error[9];memset(error,0x56,sizeof error);
    int accepted=cg_manifest_load(directory,budget,&box.value,error,8);
    expect(!accepted,label);expect(!memcmp(saved,&box,sizeof box),"failed load must preserve output/canaries");
    if(!accepted){expect(error[0]!=0&&memchr(error,0,8)!=NULL,"bounded error must be terminated");}
    expect(error[8]==0x56,"error buffer canary");
}
static void rejects(uint64_t budget,const char *label){rejects_at(root,budget,label);}

static void canonical_and_admission(void){
    char files[256];camera_files(files,sizeof files);
    camera(files,"{\"model_xml\":\"model.xml\",\"scale\":4,\"tile_overlap\":1,\"inference_precision\":\"f32\",\"max_output_pixels\":64,\"max_output_bytes\":4096}",NULL);
    CGManifest m;char error[128];memset(&m,0x41,sizeof m);strcpy(error,"stale");
    int accepted=cg_manifest_load(root,512,&m,error,sizeof error);
    expect(accepted,"canonical two-asset camera");
    if(accepted){
        expect(m.modality==CNG_CAMERA&&!strcmp(m.family,"camera_upscale")&&!strcmp(m.model_id,"camera-fixture"),"typed identity/modality");
        expect(m.asset_bytes==6&&m.base_estimated_bytes==384&&m.memory_budget_bytes==512,"independent exact SHA and admission accounting");
        expect(m.max_output_pixels==64&&m.max_output_bytes==4096&&m.tile_overlap==1&&!m.inference_f16,"camera typed options");
        expect(!strcmp(m.root,root)&&!strcmp(m.model_dir,root)&&strstr(m.sr_xml,"/model.xml")!=NULL&&!error[0],"resolved camera paths and cleared error");
    }
    rejects(383,"resident+workspace estimate over budget");
    expect(cg_manifest_load(root,384,&m,error,sizeof error),"exact admission boundary");
    rejects(5,"assets exceed budget");rejects(0,"zero budget");rejects((UINT64_C(64)<<30)+1,"overhardlimit budget");
    camera(files,"{\"model_xml\":\"model.xml\",\"inference_precision\":\"f16\"}",NULL);
    expect(cg_manifest_load(root,512,&m,error,sizeof error)&&m.inference_f16,"explicit f16 hint");
    camera(files,"{\"model_xml\":\"model.xml\"}",NULL);
    write_file("model.bin","bad",3);rejects(512,"binary hash mismatch");write_file("model.bin","bin",3);
    uint8_t large[513]={0};write_file("model.bin",large,sizeof large);rejects(512,"oversized asset before hash");write_file("model.bin","bin",3);
    write_file("model.bin","",0);rejects(512,"empty declared asset");write_file("model.bin","bin",3);
}

static void json_and_options(void){
    char files[256];camera_files(files,sizeof files);
    const char *extras[]={",\"family\":\"camera_upscale\"",",\"fam\\u0069ly\":\"camera_upscale\"",
        ",\"notes\":1e999",",\"max_input_bytes\":true",",\"resident_bytes\":-1"};
    for(size_t i=0;i<sizeof extras/sizeof extras[0];i++){
        camera(files,"{\"model_xml\":\"model.xml\"}",extras[i]);rejects(512,"duplicate/nonfinite/top typed JSON");
    }
    const char *bad_options[]={
        "{\"model_xml\":\"model.xml\",\"tile_overlap\":true}",
        "{\"model_xml\":\"model.xml\",\"tile_overlap\":0.5}",
        "{\"model_xml\":\"model.xml\",\"scale\":2}",
        "{\"model_xml\":\"model.xml\",\"max_tokens\":257}",
        "{\"model_xml\":\"model.xml\",\"max_output_bytes\":true}",
        "{\"model_xml\":\"model.xml\",\"max_output_pixels\":0}",
        "{\"model_xml\":\"model.xml\",\"max_audio_seconds\":\"30\"}",
        "{\"model_xml\":\"model.xml\",\"inference_precision\":\"int8\"}",
        "{\"model_xml\":\"model.xml\",\"inference_precision\":1}",
        "{\"model_xml\":\"model.xml\",\"inference_precision\":false}",
        "{\"model_xml\":\"model.xml\",\"inference_precision\":null}",
        "{\"model_xml\":\"model.xml\",\"model_path\":17}",
        "{\"model_xml\":\"model.xml\",\"model_path\":null}",
        "{\"model_xml\":7}","[]"};
    for(size_t i=0;i<sizeof bad_options/sizeof bad_options[0];i++){
        camera(files,bad_options[i],NULL);rejects(512,bad_options[i]);
    }
    camera("{}","{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"empty file map");
    char only[128];snprintf(only,sizeof only,"{\"model.xml\":\"%s\"}",sha_xml);
    camera(only,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"missing required camera bin");
    camera(files,"{}",NULL);rejects(512,"missing trained model_xml");
    manifest("speech","super_resolution_openvino",files,"{}",NULL);rejects(512,"family/adapter mismatch");
    uint8_t *oversized=malloc(65537);assert(oversized);memset(oversized,' ',65537);write_file("colibri_runtime.json",oversized,65537);
    rejects(512,"manifest byte bound");free(oversized);
    write_file("colibri_runtime.json","{}\0junk",7);rejects(512,"raw embedded NUL");
    char byte=0x7b;write_file("colibri_runtime.json",&byte,1);rejects(512,"truncated JSON");
}

static void path_and_aliases(void){
    char base[256],files[2048];camera_files(base,sizeof base);
    const char *keys[]={"../escape.xml","/outside.xml","sub//file","sub/../file","./model.xml",
        "CON.bin","lpt1.dat","model.xml.","space ","C:/file","sub\\\\model.xml"};
    for(size_t i=0;i<sizeof keys/sizeof keys[0];i++){
        snprintf(files,sizeof files,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\",\"%s\":\"%s\"}",sha_xml,sha_bin,keys[i],sha_xml);
        camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"portable asset path escape/alias");
    }
    snprintf(files,sizeof files,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\",\"model.xml\":\"%s\"}",sha_xml,sha_bin,sha_xml);
    camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"duplicate file key");
    write_file("MODEL.XML","xml",3);
    snprintf(files,sizeof files,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\",\"MODEL.XML\":\"%s\"}",sha_xml,sha_bin,sha_xml);
    camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"case-alias duplicate asset");
#ifndef _WIN32
    unlink_file("MODEL.XML");
#endif
    /* Do not create uppercase self-manifest on Windows: its filesystem would
     * alias and overwrite the real manifest. Linux regression still proves
     * portable casefold reservation before opening that declared extra asset. */
#ifndef _WIN32
    write_file("COLIBRI_RUNTIME.JSON","xml",3);
    snprintf(files,sizeof files,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\",\"COLIBRI_RUNTIME.JSON\":\"%s\"}",sha_xml,sha_bin,sha_xml);
    camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"case-alias of reserved manifest name");unlink_file("COLIBRI_RUNTIME.JSON");
#endif
    camera(base,"{\"model_xml\":\"model.xml\"}",NULL);
    unlink_file("model.bin");hardlink_name("model.bin","model.xml");
    snprintf(files,sizeof files,"{\"model.xml\":\"%s\",\"model.bin\":\"%s\"}",sha_xml,sha_xml);
    camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"duplicate declared hardlink inode");
    unlink_file("model.bin");write_file("model.bin","bin",3);
    unlink_file("model.xml");mkdir_name("model.xml");camera(base,"{\"model_xml\":\"model.xml\"}",NULL);
    rejects(512,"declared file is a directory");rmdir_name("model.xml");write_file("model.xml","xml",3);
#ifndef _WIN32
    unlink_file("model.bin");char fifo[4096];join(fifo,sizeof fifo,root,"model.bin");assert(!mkfifo(fifo,0600));
    alarm(3);rejects(512,"declared FIFO rejected without blocking");alarm(0);unlink_file("model.bin");write_file("model.bin","bin",3);
#endif
    unlink_file("model.bin");
    if(symlink_name("model.bin","model.xml",0)){
        camera(files,"{\"model_xml\":\"model.xml\"}",NULL);rejects(512,"symlink declared asset");unlink_file("model.bin");
    }
    write_file("model.bin","bin",3);camera(base,"{\"model_xml\":\"model.xml\"}",NULL);
    if(symlink_name("alias-root","",1)){
        char alias[4096];join(alias,sizeof alias,root,"alias-root");rejects_at(alias,512,"symlink model root");
#ifdef _WIN32
        rmdir_name("alias-root");
#else
        unlink_file("alias-root");
#endif
    }
    FILE *input=open_file("colibri_runtime.json","rb");assert(input);
    char saved[4096];size_t size=fread(saved,1,sizeof saved,input);assert(size&&size<sizeof saved);assert(!fclose(input));
    write_file("saved-manifest.json",saved,size);unlink_file("colibri_runtime.json");
    if(symlink_name("colibri_runtime.json","saved-manifest.json",0)){
        rejects(512,"symlink manifest itself");unlink_file("colibri_runtime.json");
    }
    write_file("colibri_runtime.json",saved,size);unlink_file("saved-manifest.json");
}

static void required_pipelines(void){
    const char *names[]={"config.json","generation_config.json","preprocessor_config.json",
        "openvino_encoder_model.xml","openvino_encoder_model.bin","openvino_decoder_model.xml","openvino_decoder_model.bin",
        "openvino_model.xml","openvino_model.bin","openvino_tokenizer.xml","openvino_tokenizer.bin","openvino_detokenizer.xml","openvino_detokenizer.bin"};
    mkdir_name("pipeline");char files[4096];size_t used=0;files[used++]='{';
    for(size_t i=0;i<sizeof names/sizeof names[0];i++){
        char name[512];snprintf(name,sizeof name,"pipeline/%s",names[i]);write_file(name,"xml",3);
        int n=snprintf(files+used,sizeof files-used,"%s\"%s\":\"%s\"",i?",":"",name,sha_xml);assert(n>0&&(size_t)n<sizeof files-used);used+=(size_t)n;
    }
    files[used++]='}';files[used]=0;
    CGManifest m;char error[128];
    manifest("text_graph","text_openvino",files,"{\"model_path\":\"pipeline\"}",NULL);
    expect(cg_manifest_load(root,512,&m,error,sizeof error)&&m.modality==CNG_TEXT&&strstr(m.model_dir,"/pipeline"),"nested required text assets");
    manifest("speech","whisper_openvino",files,"{\"model_path\":\"pipeline\",\"max_audio_seconds\":29.5,\"max_new_tokens\":448}",NULL);
    expect(cg_manifest_load(root,512,&m,error,sizeof error)&&m.modality==CNG_AUDIO&&m.max_audio_seconds==29.5&&m.max_tokens==448,"nested required speech assets");
    unlink_file("pipeline/openvino_decoder_model.bin");rejects(512,"missing speech asset");write_file("pipeline/openvino_decoder_model.bin","xml",3);
    /* Aliased parent directory must be checked, not only leaf assets. */
    if(symlink_name("pipeline-alias","pipeline",1)){
        char renamed[4096];
        size_t n=0;renamed[n++]='{';
        for(size_t i=0;i<sizeof names/sizeof names[0];i++){
            int wrote=snprintf(renamed+n,sizeof renamed-n,"%s\"pipeline-alias/%s\":\"%s\"",i?",":"",names[i],sha_xml);
            assert(wrote>0&&(size_t)wrote<sizeof renamed-n);n+=(size_t)wrote;
        }
        renamed[n++]='}';renamed[n]=0;
        manifest("speech","whisper_openvino",renamed,"{\"model_path\":\"pipeline-alias\"}",NULL);rejects(512,"symlink parent directory");
#ifdef _WIN32
        rmdir_name("pipeline-alias");
#else
        unlink_file("pipeline-alias");
#endif
    }
    for(size_t i=0;i<sizeof names/sizeof names[0];i++){char name[512];snprintf(name,sizeof name,"pipeline/%s",names[i]);unlink_file(name);}
    rmdir_name("pipeline");
}

int main(void){
    setup();canonical_and_admission();json_and_options();path_and_aliases();required_pipelines();cleanup();
    if(failures){fprintf(stderr,"native manifest: %u failed of %u checks\n",failures,checks);return 1;}
    printf("native manifest: %u SDK-free identity, path/alias, typed-options, admission and transactional checks passed\n",checks);return 0;
}
