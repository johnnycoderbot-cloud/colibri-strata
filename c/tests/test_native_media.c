/* Native media gates. Pillow12.1.0 generated the independent bicubic fixture
 * hashes below; these tests run with pure C only and never execute a model. */
#include "../native_media.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put16(uint8_t *p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);}
static void put32(uint8_t *p,uint32_t v){for(int i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
static uint8_t *wav(size_t count,int floating,size_t *bytes){
    size_t unit=floating?4:2;*bytes=44+count*unit;
    uint8_t *p=calloc(*bytes,1);assert(p);memcpy(p,"RIFF",4);put32(p+4,(uint32_t)(*bytes-8));
    memcpy(p+8,"WAVEfmt ",8);put32(p+16,16);put16(p+20,floating?3:1);put16(p+22,1);
    put32(p+24,16000);put32(p+28,(uint32_t)(16000*unit));put16(p+32,(uint16_t)unit);
    put16(p+34,floating?32:16);memcpy(p+36,"data",4);put32(p+40,(uint32_t)(count*unit));return p;
}
static void check_bad_wav(const uint8_t *p,size_t n,size_t max_bytes,double seconds){
    nm_wav_info info,before;memset(&info,0x35,sizeof info);memcpy(&before,&info,sizeof info);
    float out[16],saved[16];for(int i=0;i<16;i++)out[i]=saved[i]=17.25f;
    assert(nm_wav_inspect(p,n,max_bytes,seconds,&info)!=NM_OK);
    assert(!memcmp(&info,&before,sizeof info));
    assert(nm_wav_decode(p,n,max_bytes,seconds,out,16,&info)!=NM_OK);
    assert(!memcmp(&info,&before,sizeof info));assert(!memcmp(out,saved,sizeof out));
}
static void wav_tests(void){
    size_t n;uint8_t *p=wav(5,0,&n);int16_t values[]={-32768,-16384,0,16384,32767};
    for(int i=0;i<5;i++)put16(p+44+i*2,(uint16_t)values[i]);
    nm_wav_info info;float out[7]={-17,0,0,0,0,0,19};
    assert(nm_wav_decode(p,n,NM_WAV_DEFAULT_BYTES,30,out+1,5,&info)==NM_OK);
    const float expected[]={-1,-.5f,0,.5f,32767.f/32768.f};
    assert(!memcmp(out+1,expected,sizeof expected)&&out[0]==-17&&out[6]==19);
    assert(info.sample_count==5&&info.sample_rate==16000&&info.channels==1&&info.encoding==NM_WAV_PCM16);
    assert(info.duration_seconds==5.0/16000.0);
    nm_wav_info sentinel;memset(&sentinel,0xa5,sizeof sentinel);info=sentinel;
    float saved[7];memcpy(saved,out,sizeof out);
    assert(nm_wav_decode(p,n,NM_WAV_DEFAULT_BYTES,30,out+1,4,&info)==NM_BOUNDS);
    assert(!memcmp(out,saved,sizeof out)&&!memcmp(&info,&sentinel,sizeof info));
    assert(nm_wav_decode(p,n,NM_WAV_DEFAULT_BYTES,30,(float *)p,5,&info)==NM_ALIAS);
    assert(nm_wav_inspect(p,n,NM_WAV_DEFAULT_BYTES,30,(nm_wav_info *)p)==NM_ALIAS);
    uint8_t *unaligned=malloc(n+1);assert(unaligned);memcpy(unaligned+1,p,n);
    assert(nm_wav_decode(unaligned+1,n,NM_WAV_DEFAULT_BYTES,30,out+1,5,&info)==NM_OK);free(unaligned);
    check_bad_wav(p,n-1,NM_WAV_DEFAULT_BYTES,30);check_bad_wav(p,n,n-1,30);
    check_bad_wav(p,n,0,30);check_bad_wav(p,n,NM_WAV_MAX_BYTES+1,30);
    check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,NAN);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,INFINITY);
    check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,0);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,30.001);
    uint8_t *bad=malloc(n+24);assert(bad);
    const int offsets[]={0,4,8,22,24,28,32,34};
    for(size_t k=0;k<sizeof offsets/sizeof offsets[0];k++){
        memcpy(bad,p,n);bad[offsets[k]]^=0x55;check_bad_wav(bad,n,NM_WAV_DEFAULT_BYTES,30);
    }
    /* Unknown odd chunk is accepted only with canonical zero padding. */
    memcpy(bad,p,12);memcpy(bad+12,"JUNK",4);put32(bad+16,1);bad[20]=73;bad[21]=0;
    memcpy(bad+22,p+12,n-12);put32(bad+4,(uint32_t)(n+10-8));
    assert(nm_wav_inspect(bad,n+10,NM_WAV_DEFAULT_BYTES,30,&info)==NM_OK);
    bad[21]=1;check_bad_wav(bad,n+10,NM_WAV_DEFAULT_BYTES,30);
    memcpy(bad,p,36);memcpy(bad+36,p+12,24);memcpy(bad+60,p+36,n-36);put32(bad+4,(uint32_t)(n+24-8));
    check_bad_wav(bad,n+24,NM_WAV_DEFAULT_BYTES,30);free(bad);free(p);

    p=wav(6,1,&n);const uint32_t bits[]={0xbf800000u,0x80000000u,0u,0x3f000000u,0x3f800000u,1u};
    for(int i=0;i<6;i++)put32(p+44+i*4,bits[i]);
    float f[6];assert(nm_wav_decode(p,n,NM_WAV_DEFAULT_BYTES,30,f,6,&info)==NM_OK);
    for(int i=0;i<6;i++){uint32_t value;memcpy(&value,f+i,4);assert(value==bits[i]);}
    const uint32_t invalid[]={0x7fc00000u,0x7f800000u,0xff800000u,0x3f800001u,0xbf800001u};
    for(size_t i=0;i<sizeof invalid/sizeof invalid[0];i++){
        put32(p+44+5*4,invalid[i]);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,30);
    }
    free(p);p=wav(0,0,&n);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,30);free(p);
    p=wav(479999,0,&n);assert(nm_wav_inspect(p,n,NM_WAV_DEFAULT_BYTES,30,&info)==NM_OK);
    assert(info.sample_count==479999);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,.1);free(p);
    p=wav(480000,0,&n);check_bad_wav(p,n,NM_WAV_DEFAULT_BYTES,30);free(p);
}

static uint64_t fnv_byte(uint64_t h,uint8_t value){return (h^value)*UINT64_C(1099511628211);}
static void resize_goldens(void){
    static const struct {size_t w,h,c;uint64_t hash;} cases[]={
        {1,1,1,UINT64_C(0x82bcc1db63088ed5)},{1,1,3,UINT64_C(0xadc8599026600ba5)},
        {1,7,1,UINT64_C(0x8e215a4b7e80a655)},{9,1,3,UINT64_C(0x15ddc1b2a76f1acd)},
        {2,2,1,UINT64_C(0x72929a9cd8291565)},{3,2,3,UINT64_C(0xa6bd915639cb287c)},
        {7,5,3,UINT64_C(0x30836a16c1406514)},{11,9,1,UINT64_C(0x5baa3db26ce92e5d)},
        {19,17,3,UINT64_C(0x66ee6741261026fc)},{33,31,1,UINT64_C(0x0ad104edc9beef9c)}
    };
    for(size_t k=0;k<sizeof cases/sizeof cases[0];k++){
        size_t w=cases[k].w,h=cases[k].h,c=cases[k].c,ss=w*c+3,ds=w*4*c+5,work_bytes=0;
        assert(w<=33&&h<=31&&c<=3);
        uint8_t *src=malloc(h*ss),*out=malloc(h*4*ds+7);assert(src&&out);
        memset(src,0x7b,h*ss);memset(out,0xa9,h*4*ds+7);
        for(size_t y=0;y<h;y++)for(size_t x=0;x<w*c;x++){
            size_t i=y*w*c+x;src[y*ss+x]=(uint8_t)((i*37+(i/11)*19+23)%256);
        }
        assert(nm_bicubic4_workspace_bytes(w,h,c,&work_bytes)==NM_OK);
        assert(work_bytes==w*4*h*c);uint8_t *wb=malloc(work_bytes+6);assert(wb);memset(wb,0x67,work_bytes+6);
        assert(nm_bicubic4_u8(src,h*ss,w,h,c,ss,out,h*4*ds,ds,wb+3,work_bytes)==NM_OK);
        uint64_t hash=UINT64_C(14695981039346656037);
        for(size_t y=0;y<h*4;y++){
            for(size_t x=0;x<w*4*c;x++)hash=fnv_byte(hash,out[y*ds+x]);
            for(size_t x=w*4*c;x<ds;x++)assert(out[y*ds+x]==0xa9);
        }
        if(hash!=cases[k].hash)fprintf(stderr,"Pillow fixture mismatch %zux%zu c%zu: %016llx != %016llx\n",w,h,c,(unsigned long long)hash,(unsigned long long)cases[k].hash);
        assert(hash==cases[k].hash);
        assert(wb[0]==0x67&&wb[2]==0x67&&wb[work_bytes+3]==0x67);
        for(size_t i=h*4*ds;i<h*4*ds+7;i++)assert(out[i]==0xa9);
        memset(out,0x35,h*4*ds);memset(wb,0x67,work_bytes+6);
        assert(nm_bicubic4_u8(src,h*ss,w,h,c,ss,out,h*4*ds,ds,wb+3,work_bytes-1)==NM_BOUNDS);
        for(size_t i=0;i<h*4*ds;i++)assert(out[i]==0x35);
        for(size_t i=0;i<work_bytes+6;i++)assert(wb[i]==0x67);
        assert(nm_bicubic4_u8(src,h*ss,w,h,c,ss,src,h*ss,ss,wb+3,work_bytes)!=NM_OK);
        free(src);free(out);free(wb);
    }
    size_t sentinel=719;
    assert(nm_bicubic4_workspace_bytes(SIZE_MAX,2,3,&sentinel)==NM_BOUNDS&&sentinel==719);
    assert(nm_bicubic4_workspace_bytes(1,1,4,&sentinel)==NM_UNSUPPORTED&&sentinel==719);
    assert(nm_bicubic4_workspace_bytes(0,1,1,&sentinel)==NM_INVALID&&sentinel==719);
}

static void sr_prepare_tests(void){
    uint8_t src[28];memset(src,0x55,sizeof src);
    for(size_t y=0;y<2;y++)for(size_t x=0;x<3*4;x++){
        size_t i=y*3*4+x;src[y*14+x]=(uint8_t)((i*37+(i/11)*19+23)%256);
    }
    float low[27+2],high[432+2];low[0]=-41;low[28]=43;high[0]=-47;high[433]=53;
    size_t wb=0;assert(nm_sr_workspace_bytes(3,3,&wb)==NM_OK&&wb==135);
    uint8_t work[135+2];memset(work,0x65,sizeof work);
    assert(nm_sr_prepare(src,sizeof src,3,2,4,14,-1,-1,3,3,low+1,27,high+1,432,work+1,135)==NM_OK);
    for(size_t y=0;y<3;y++)for(size_t x=0;x<3;x++)for(size_t c=0;c<3;c++){
        size_t sy=y?y-1:0,sx=x?x-1:0;
        assert(low[1+c*9+y*3+x]==(float)src[sy*14+sx*4+(2-c)]);
    }
    uint64_t hash=UINT64_C(14695981039346656037);
    for(int i=0;i<432;i++){assert(high[i+1]>=0&&high[i+1]<=255);hash=fnv_byte(hash,(uint8_t)high[i+1]);}
    assert(hash==UINT64_C(0x47f359a482dd5315));
    assert(low[0]==-41&&low[28]==43&&high[0]==-47&&high[433]==53&&work[0]==0x65&&work[136]==0x65);
    float saved_low[29],saved_high[434];memcpy(saved_low,low,sizeof low);memcpy(saved_high,high,sizeof high);
    uint8_t saved_work[137];memcpy(saved_work,work,sizeof work);
    assert(nm_sr_prepare(src,sizeof src,3,2,4,14,0,0,3,3,low+1,26,high+1,432,work+1,135)==NM_BOUNDS);
    assert(nm_sr_prepare(src,sizeof src,3,2,4,14,INT64_MAX,0,3,3,low+1,27,high+1,432,work+1,135)==NM_BOUNDS);
    assert(nm_sr_prepare(src,sizeof src,3,2,4,14,0,0,3,3,low+1,27,high+1,432,work+1,134)==NM_BOUNDS);
    assert(nm_sr_prepare(src,sizeof src,3,2,4,14,0,0,3,3,low+1,27,low+1,432,work+1,135)==NM_ALIAS);
    assert(!memcmp(saved_low,low,sizeof low)&&!memcmp(saved_high,high,sizeof high)&&!memcmp(saved_work,work,sizeof work));
    assert(nm_sr_workspace_bytes(1024,1024,&wb)==NM_BOUNDS);
    assert(nm_sr_workspace_bytes(480,270,&wb)==NM_OK&&wb==480*270*15);
    /* A one-pixel RGBA source extends constantly, with alpha ignored. */
    uint8_t single[]={13,29,101,0};
    assert(nm_sr_prepare(single,4,1,1,4,4,-7,-11,3,3,low+1,27,high+1,432,work+1,135)==NM_OK);
    for(int c=0;c<3;c++)for(int p=0;p<144;p++)assert(high[1+c*144+p]==single[2-c]);
}

static void scores_tests(void){
    float scores[12]={-1,.25f,.5f,2, .25f,.5f,1,0, 2,.25f,-.5f,.5f};
    uint8_t out[16];memset(out,0x65,sizeof out);
    assert(nm_bgr_output_rgb(scores,12,2,2,out,sizeof out,8)==NM_OK);
    const uint8_t expected[]={255,63,0,63,127,63, 0,255,127,127,0,255};
    assert(!memcmp(out,expected,6)&&!memcmp(out+8,expected+6,6));
    assert(out[6]==0x65&&out[7]==0x65&&out[14]==0x65&&out[15]==0x65);
    memset(out,0x39,sizeof out);scores[11]=NAN;
    assert(nm_bgr_output_rgb(scores,12,2,2,out,sizeof out,8)==NM_INVALID);
    for(size_t i=0;i<sizeof out;i++)assert(out[i]==0x39);
    scores[11]=INFINITY;assert(nm_bgr_output_rgb(scores,12,2,2,out,sizeof out,8)==NM_INVALID);
    scores[11]=.5f;
    assert(nm_bgr_output_rgb(scores,11,2,2,out,sizeof out,8)==NM_BOUNDS);
    assert(nm_bgr_output_rgb(scores,12,SIZE_MAX,2,out,sizeof out,8)==NM_BOUNDS);
    assert(nm_bgr_output_rgb(scores,12,2,2,(uint8_t *)scores,sizeof scores,8)==NM_ALIAS);
    scores[0]=FLT_MAX;scores[4]=-FLT_MAX;
    assert(nm_bgr_output_rgb(scores,12,2,2,out,sizeof out,8)==NM_OK&&out[2]==255&&out[1]==0);
}

int main(void){
    wav_tests();resize_goldens();sr_prepare_tests();scores_tests();
    assert(!strcmp(nm_status_string(NM_OK),"ok"));
    puts("native media: WAV transactional bounds, Pillow bicubic/alpha, edge/BGR preparation and model-score truncation exact");
    return 0;
}
