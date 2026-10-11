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
int main(void){char directory[1024];int result;
#ifdef _WIN32
    char temp[MAX_PATH];CHECK(GetTempPathA(MAX_PATH,temp));CHECK(GetTempFileNameA(temp,"snh",0,directory));
    CHECK(DeleteFileA(directory));CHECK(!_mkdir(directory));
#else
    strcpy(directory,"/tmp/strata-native-heads-XXXXXX");CHECK(mkdtemp(directory));
#endif
    result=fingerprint_tests(directory)||pack_tests(directory);cleanup(directory);
    if(!result)puts("native heads: identity, tamper, canonical storage, transactional AVX/scalar/parallel tests passed");
    return result;
}
