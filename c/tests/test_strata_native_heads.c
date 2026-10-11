#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include "../strata_native_heads.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

#define CHECK(c) do {if(!(c)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#c);return 1;}}while(0)
/* Independent protocol fixture: two rank17 identity projections (last latent
 * also reads h[0]), nine one-hot score rows, head1 token1 bias20. It exercises
 * AVX tails and odd INT4 padding. Expected tokens follow from this description.
 * Digest constants were computed with Python's hashlib, not the loader. */
static const char good_hex[] =
"43534e480100000060000000110000000900000002000000040000000000000011111111111111111111111111111111111111111111111111111111111111118b8e4906522f1018b1fe9fe002219c384d7004061bdb0d257d5c4288f37a26660200000011000000bd00000000000000710000000000000009000000000000004353513101000000040000002101000040000000050000000000803f07000000000000000007000000000000000007000000000000000007000000000000803f00000000070000000000000000070000000000000000070000000000000000070000803f00000000000000000700000000000000000700000000000000000700000000000000803f00000007000000000000000007000000000000000007000000000000000007000000803f00000000000000070700000000000000074353513101000000040000009900000040000000030000000000803f07000000000000000007000000000000000007000000000000000007000000000000803f00000000070000000000000000070000000000000000070000000000000000070000803f000000000000000007000000000000000000000000000000000000000000000000000000000000000000000000000000000300000011000000bd00000000000000710000000000000009000000000000004353513101000000040000002101000040000000050000000000803f07000000000000000007000000000000000007000000000000000007000000000000803f00000000070000000000000000070000000000000000070000000000000000070000803f00000000000000000700000000000000000700000000000000000700000000000000803f00000007000000000000000007000000000000000007000000000000000007000000803f00000000000000070700000000000000074353513101000000040000009900000040000000030000000000803f07000000000000000007000000000000000007000000000000000007000000000000803f00000000070000000000000000070000000000000000070000000000000000070000803f00000000000000000700000000000000000000a04100000000000000000000000000000000000000000000000000000000";

static unsigned nibble(char c){return c<='9'?(unsigned)(c-'0'):(unsigned)(c-'a'+10);}
static void unhex(uint8_t *out,const char *hex,size_t n){size_t i;for(i=0;i<n;i++)out[i]=(uint8_t)(nibble(hex[2*i])*16+nibble(hex[2*i+1]));}
static int path_for(char out[1024],const char *dir,const char *name){size_t a=strlen(dir),b=strlen(name);
    if(a>=1024||b>=1024||a+b+2>1024)return 0;
    memcpy(out,dir,a);out[a]='/';memcpy(out+a+1,name,b+1);return 1;
}
static int write_file(const char *dir,const char *name,const void *data,size_t size){char path[1024];FILE *f;
    if(!path_for(path,dir,name))return 0;
    f=fopen(path,"wb");if(!f)return 0;
    if(fwrite(data,1,size,f)!=size){fclose(f);return 0;}return fclose(f)==0;
}
static void cleanup(const char *dir){const char *names[]={"bank.snh","config.json","tokenizer.json","qwen36_meta.json","a.safetensors","z.safetensors","b.safetensors","ignored.txt"};
    char path[1024];size_t i;for(i=0;i<sizeof(names)/sizeof(names[0]);i++)if(path_for(path,dir,names[i]))remove(path);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif
}
static int fingerprint_tests(const char *dir){uint8_t fp[32],expected[32],changed[32];char from[1024],to[1024];
    CHECK(write_file(dir,"z.safetensors","zzz",3));CHECK(write_file(dir,"config.json","{}",2));
    CHECK(write_file(dir,"tokenizer.json","tok",3));CHECK(write_file(dir,"a.safetensors","aaa",3));
    CHECK(write_file(dir,"qwen36_meta.json","meta",4));
    unhex(expected,"bedf3db495dcdf0afcf3871eac6a1280eae68d9dfe7f5d0d1b9a64211d532c75",32);
    CHECK(sn_fingerprint_model(dir,"fixture-native-f32",fp)==SN_OK);
    CHECK(!memcmp(fp,expected,32));
    CHECK(write_file(dir,"ignored.txt","ignored",7));CHECK(sn_fingerprint_model(dir,"fixture-native-f32",changed)==SN_OK);
    CHECK(!memcmp(fp,changed,32));
    CHECK(sn_fingerprint_model(dir,"fixture-native-int8",changed)==SN_OK&&memcmp(fp,changed,32));
    CHECK(write_file(dir,"config.json","{ }",3));CHECK(sn_fingerprint_model(dir,"fixture-native-f32",changed)==SN_OK&&memcmp(fp,changed,32));
    CHECK(write_file(dir,"config.json","{}",2));
    CHECK(path_for(from,dir,"a.safetensors"));CHECK(path_for(to,dir,"b.safetensors"));CHECK(!rename(from,to));
    CHECK(sn_fingerprint_model(dir,"fixture-native-f32",changed)==SN_OK&&memcmp(fp,changed,32));CHECK(!rename(to,from));
    memset(changed,0x5a,32);CHECK(sn_fingerprint_model(dir,"",changed)==SN_INVALID);CHECK(changed[0]==0x5a);
    return 0;
}
static int pack_tests(const char *dir){uint8_t good[836],bad[837],fp[32],other[32];char path[1024];
    sn_bank *bank=NULL,*rejected=NULL;sn_info info;float hidden[17];int tokens[2],serial[2],i;
    CHECK(strlen(good_hex)==sizeof(good)*2);unhex(good,good_hex,sizeof(good));memset(fp,17,32);memset(other,18,32);
    CHECK(path_for(path,dir,"bank.snh"));CHECK(write_file(dir,"bank.snh",good,sizeof(good)));
#ifndef _WIN32
    {char fifo[1024];CHECK(path_for(fifo,dir,"fifo.snh"));CHECK(!mkfifo(fifo,0600));
     alarm(3);CHECK(sn_open(fifo,fp,17,9,&rejected)==SN_IO&&!rejected);alarm(0);CHECK(!unlink(fifo));}
#endif
    CHECK(sn_open(dir,fp,17,9,&rejected)==SN_IO&&!rejected);
    CHECK(sn_open(path,fp,17,9,&bank)==SN_OK);CHECK(bank);CHECK(sn_get_info(bank,&info)==SN_OK);
    CHECK(sn_bank_kind(bank)==SN_KIND_ORDINARY&&sn_bank_kind(NULL)==SN_KIND_INVALID);
    {float base[9]={0};CHECK(sn_predict_residual(bank,hidden,base,tokens)==SN_INVALID);}
    CHECK(info.hidden_dim==17&&info.vocab_size==9&&info.head_count==2&&info.source_bits==4&&info.decoded_bytes==3608);
    CHECK(!strcmp(info.kernel,"avx2")||!strcmp(info.kernel,"scalar"));
#ifdef SN_FORCE_SCALAR
    CHECK(!strcmp(info.kernel,"scalar"));
#endif
    for(i=0;i<17;i++)hidden[i]=(float)i;
    for(i=0;i<100;i++){
        CHECK(sn_predict_sized(bank,hidden,17,serial,2,0)==SN_OK);
        CHECK(sn_predict(bank,hidden,tokens)==SN_OK);CHECK(tokens[0]==8&&tokens[1]==1);
        CHECK(!memcmp(tokens,serial,sizeof(tokens)));
    }
    memset(hidden,0,sizeof(hidden));CHECK(sn_predict(bank,hidden,tokens)==SN_OK&&tokens[0]==0&&tokens[1]==1);
    tokens[0]=111;tokens[1]=222;hidden[0]=NAN;CHECK(sn_predict(bank,hidden,tokens)==SN_NONFINITE);
    CHECK(tokens[0]==111&&tokens[1]==222);hidden[0]=0;
    CHECK(sn_predict_sized(bank,hidden,16,tokens,2,1)==SN_DIMENSIONS);
    CHECK(sn_predict_sized(bank,hidden,17,tokens,1,1)==SN_DIMENSIONS);
    CHECK(sn_predict_sized(bank,hidden,17,(int *)hidden,2,1)==SN_INVALID);
    for(i=0;i<17;i++)hidden[i]=FLT_MAX;
    CHECK(sn_predict(bank,hidden,tokens)==SN_OVERFLOW&&tokens[0]==111&&tokens[1]==222);
    CHECK(sn_open(path,fp,17,9,&bank)==SN_INVALID); /* Existing bank ownership stays intact. */
    CHECK(sn_open(path,other,17,9,&rejected)==SN_IDENTITY&&!rejected);
    CHECK(sn_open(path,fp,16,9,&rejected)==SN_DIMENSIONS&&!rejected);
    memcpy(bad,good,sizeof(good));bad[24]=8;CHECK(write_file(dir,"bank.snh",bad,sizeof(good)));
    CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
    memcpy(bad,good,sizeof(good));bad[500]^=1;CHECK(write_file(dir,"bank.snh",bad,sizeof(good)));
    CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
    memcpy(bad,good,sizeof(good));bad[836]=0;CHECK(write_file(dir,"bank.snh",bad,sizeof(bad)));
    CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
    CHECK(write_file(dir,"bank.snh",good,95));CHECK(sn_open(path,fp,17,9,&rejected)==SN_LIMIT&&!rejected);
    memcpy(bad,good,sizeof(good));bad[430]=0;bad[431]=0;bad[432]=0xc0;bad[433]=0x7f;
    unhex(bad+64,"dfc9ba3bb7955e69d29c8735c4ac4912cda3bf64c37cbb6b7685d6aaf079395e",32);
    CHECK(write_file(dir,"bank.snh",bad,sizeof(good)));CHECK(sn_open(path,fp,17,9,&rejected)==SN_NONFINITE&&!rejected);
    memcpy(bad,good,sizeof(good));bad[466]=4;
    unhex(bad+64,"086a37ac1c1a5036c14d54e5db21b90d8f90764a2082141294ff42aa435f47d1",32);
    CHECK(write_file(dir,"bank.snh",bad,sizeof(good)));CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
    memcpy(bad,good,sizeof(good));bad[316]|=16;
    unhex(bad+64,"81b4e49521c85f9f6c62d41b047fd566f357b8f5be453b33990cf24fd10d2549",32);
    CHECK(write_file(dir,"bank.snh",bad,sizeof(good)));CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
    /* A loaded bank owns its decoded weights: changing the source file does not
     * mutate proposals or trigger per-call reads/weight validation. */
    for(i=0;i<17;i++)hidden[i]=(float)i;
    CHECK(sn_predict(bank,hidden,tokens)==SN_OK&&tokens[0]==8&&tokens[1]==1);
    sn_close(bank);sn_close(NULL);return 0;
}
static int residual_tests(const char *dir){
    uint8_t bytes[836],fp[32];char path[1024];sn_bank *bank=NULL,*rejected=NULL;
    float h[17],base[9],before[9];int tokens[2],serial[2],i,mode;
    unhex(bytes,good_hex,sizeof(bytes));memset(fp,17,32);bytes[4]=2;bytes[28]=1;
    CHECK(path_for(path,dir,"bank.snh"));CHECK(write_file(dir,"bank.snh",bytes,sizeof(bytes)));
    CHECK(sn_open(path,fp,17,9,&bank)==SN_OK&&sn_bank_kind(bank)==SN_KIND_LOGIT_RESIDUAL);
    tokens[0]=111;tokens[1]=222;
    CHECK(sn_predict(bank,h,tokens)==SN_INVALID);
    CHECK(sn_predict_sized(bank,h,17,tokens,2,0)==SN_INVALID&&tokens[0]==111&&tokens[1]==222);
    for(i=0;i<17;i++)h[i]=(float)i;
    memset(base,0,sizeof(base));base[4]=100.f;memcpy(before,base,sizeof(base));
    for(i=0;i<30;i++){
        CHECK(sn_predict_residual_sized(bank,h,17,base,9,serial,2,0)==SN_OK);
        CHECK(sn_predict_residual(bank,h,base,tokens)==SN_OK);
        CHECK(tokens[0]==4&&tokens[1]==4&&!memcmp(serial,tokens,sizeof(tokens)));
        CHECK(!memcmp(base,before,sizeof(base)));
    }
    memset(h,0,sizeof(h));memset(base,0,sizeof(base));
    CHECK(sn_predict_residual(bank,h,base,tokens)==SN_OK&&tokens[0]==0&&tokens[1]==1);
    base[0]=base[2]=10.f;
    CHECK(sn_predict_residual(bank,h,base,tokens)==SN_OK&&tokens[0]==0&&tokens[1]==1);
    tokens[0]=111;tokens[1]=222;
    CHECK(sn_predict_residual_sized(bank,h,17,base,8,tokens,2,1)==SN_DIMENSIONS);
    CHECK(sn_predict_residual_sized(bank,h,17,base,10,tokens,2,1)==SN_DIMENSIONS);
    CHECK(sn_predict_residual_sized(bank,h,16,base,9,tokens,2,1)==SN_DIMENSIONS);
    CHECK(sn_predict_residual_sized(bank,h,17,base,9,tokens,1,1)==SN_DIMENSIONS);
    CHECK(sn_predict_residual(bank,h,NULL,tokens)==SN_INVALID);
    CHECK(sn_predict_residual(bank,h,base,(int *)h)==SN_INVALID);
    CHECK(sn_predict_residual(bank,h,base,(int *)(base+3))==SN_INVALID);
    CHECK(tokens[0]==111&&tokens[1]==222);
    for(i=0;i<9;i++){
        float old=base[i];base[i]=NAN;
        CHECK(sn_predict_residual(bank,h,base,tokens)==SN_NONFINITE);
        CHECK(tokens[0]==111&&tokens[1]==222);base[i]=old;
    }
    base[8]=INFINITY;CHECK(sn_predict_residual(bank,h,base,tokens)==SN_NONFINITE);base[8]=0;
    h[16]=NAN;CHECK(sn_predict_residual(bank,h,base,tokens)==SN_NONFINITE);h[16]=0;
    h[0]=FLT_MAX;base[0]=FLT_MAX;
    for(mode=0;mode<=1;mode++){
        CHECK(sn_predict_residual_sized(bank,h,17,base,9,tokens,2,mode)==SN_OVERFLOW);
        CHECK(tokens[0]==111&&tokens[1]==222);
    }
    sn_close(bank);bank=NULL;
    /* Failure in only the last head cannot publish the first head's result.
     * Independent SHA from hashlib: second head's token1 bias is FLT_MAX. */
    bytes[804]=0xff;bytes[805]=0xff;bytes[806]=0x7f;bytes[807]=0x7f;
    unhex(bytes+64,"0d729d29cfdb635d0ef947e11b92cba9346ee58a94115dfc7b9857309f1c0495",32);
    CHECK(write_file(dir,"bank.snh",bytes,sizeof(bytes)));
    CHECK(sn_open(path,fp,17,9,&bank)==SN_OK);
    memset(h,0,sizeof(h));memset(base,0,sizeof(base));base[1]=FLT_MAX;
    for(mode=0;mode<=1;mode++){
        CHECK(sn_predict_residual_sized(bank,h,17,base,9,tokens,2,mode)==SN_OVERFLOW);
        CHECK(tokens[0]==111&&tokens[1]==222);
    }
    /* No premature F32 rounding: an above-F32 delta can be cancelled by the
     * finite negative base before the final checked score conversion. */
    h[1]=FLT_MAX;base[1]=-FLT_MAX;
    CHECK(sn_predict_residual(bank,h,base,tokens)==SN_OK&&tokens[0]==0&&tokens[1]==1);
    sn_close(bank);bank=NULL;
    /* All version/kind pairings except (1,0),(2,1) must refuse before publish. */
    {const unsigned versions[]={1,2,2,3};const unsigned kinds[]={1,0,2,1};
     for(i=0;i<4;i++){
        bytes[4]=(uint8_t)versions[i];bytes[28]=(uint8_t)kinds[i];
        CHECK(write_file(dir,"bank.snh",bytes,sizeof(bytes)));
        CHECK(sn_open(path,fp,17,9,&rejected)==SN_INVALID&&!rejected);
     }}
    return 0;
}
static int replay_u32(FILE *f,uint32_t *out){uint8_t b[4];
    if(fread(b,1,4,f)!=4)return 0;
    *out=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;return 1;
}
static int replay_floats(FILE *f,float *out,size_t count){size_t i;uint32_t bits;
    for(i=0;i<count;i++){if(!replay_u32(f,&bits))return 0;memcpy(out+i,&bits,4);}return 1;
}
static int residual_replay(const char *pack_path,const char *fixture){
    FILE *f=fopen(fixture,"rb");uint8_t magic[4],fp[32];uint32_t version,H,V,N,K,i,j,word;
    sn_bank *bank=NULL;sn_info info;float *hidden,*base;int tokens[SN_MAX_HEADS],expected[SN_MAX_HEADS],mode;
    CHECK(f);CHECK(fread(magic,1,4,f)==4&&!memcmp(magic,"CSRF",4));
    CHECK(replay_u32(f,&version)&&version==1&&replay_u32(f,&H)&&replay_u32(f,&V)&&replay_u32(f,&N)&&replay_u32(f,&K));
    CHECK(H&&H<=32768&&V&&V<=1048576&&N&&N<=1024&&K&&K<=SN_MAX_HEADS);
    CHECK((uint64_t)N*((uint64_t)H+V+K)*4<=SN_MAX_FILE_BYTES);
    CHECK(fread(fp,1,32,f)==32);CHECK(sn_open(pack_path,fp,H,V,&bank)==SN_OK);
    CHECK(sn_bank_kind(bank)==SN_KIND_LOGIT_RESIDUAL&&sn_get_info(bank,&info)==SN_OK&&info.head_count==K);
    hidden=malloc((size_t)H*sizeof(float));base=malloc((size_t)V*sizeof(float));CHECK(hidden&&base);
    for(i=0;i<N;i++){
        CHECK(replay_floats(f,hidden,H)&&replay_floats(f,base,V));
        for(j=0;j<K;j++){CHECK(replay_u32(f,&word)&&word<V);expected[j]=(int)word;}
        for(mode=0;mode<=1;mode++){
            for(j=0;j<K;j++)tokens[j]=-9;
            CHECK(sn_predict_sized(bank,hidden,H,tokens,K,mode)==SN_INVALID);
            CHECK(sn_predict_residual_sized(bank,hidden,H,base,V,tokens,K,mode)==SN_OK);
            CHECK(!memcmp(tokens,expected,(size_t)K*sizeof(int)));
        }
    }
    CHECK(fgetc(f)==EOF&&!ferror(f));fclose(f);free(hidden);free(base);sn_close(bank);
    printf("native residual replay: %u captured rows, scalar/OpenMP predictions match reference\n",N);return 0;
}
int main(int argc,char **argv){char directory[1024];int result;
    if(argc==4&&!strcmp(argv[1],"--residual-replay"))return residual_replay(argv[2],argv[3]);
#ifdef _WIN32
    char temp[MAX_PATH];CHECK(GetTempPathA(MAX_PATH,temp));CHECK(GetTempFileNameA(temp,"snh",0,directory));
    CHECK(DeleteFileA(directory));CHECK(!_mkdir(directory));
#else
    strcpy(directory,"/tmp/strata-native-heads-XXXXXX");CHECK(mkdtemp(directory));
#endif
    result=fingerprint_tests(directory)||pack_tests(directory)||residual_tests(directory);cleanup(directory);
    if(!result)puts("native heads: identity, tamper, canonical storage, transactional AVX/scalar/parallel tests passed");
    return result;
}
