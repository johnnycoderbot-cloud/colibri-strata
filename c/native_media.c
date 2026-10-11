#include "native_media.h"
#include <float.h>
#include <limits.h>
#include <string.h>

static int nm_overlap(const void *a,size_t an,const void *b,size_t bn){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!an||!bn)return 0;
    if(!a||!b||an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)return 1;
    return x<=y?y-x<an:x-y<bn;
}
static int nm_aligned_float(const void *p){
    struct nm_float_alignment {char byte;float value;};
    return p && (uintptr_t)p%offsetof(struct nm_float_alignment,value)==0;
}
static int nm_aligned_info(const void *p){
    struct nm_info_alignment {char byte;nm_wav_info value;};
    return p && (uintptr_t)p%offsetof(struct nm_info_alignment,value)==0;
}
static uint16_t nm_u16(const uint8_t *p){return (uint16_t)((uint16_t)p[0]|(uint16_t)p[1]<<8);}
static uint32_t nm_u32(const uint8_t *p){
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static int nm_ieee_f32(void){return sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128;}
static int nm_finite_f32(float value){uint32_t u;memcpy(&u,&value,4);return (u&0x7f800000u)!=0x7f800000u;}
static int nm_valid_duration(double seconds){
    if(sizeof(double)!=8||DBL_MANT_DIG!=53||DBL_MAX_EXP!=1024)return 0;
    uint64_t bits;memcpy(&bits,&seconds,8);
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000) && seconds>0 && seconds<=30;
}
const char *nm_status_string(nm_status status){
    switch(status){case NM_OK:return "ok";case NM_INVALID:return "invalid media";
        case NM_BOUNDS:return "media capacity or dimension limit";case NM_ALIAS:return "overlapping media buffers";
        case NM_UNSUPPORTED:return "unsupported media format";default:return "unknown media status";}
}

static nm_status nm_wav_parse(const uint8_t *p,size_t bytes,size_t max_bytes,double seconds,
                               nm_wav_info *info,size_t *data_offset){
    if(!p||!info||!data_offset||!max_bytes||max_bytes>NM_WAV_MAX_BYTES||!nm_valid_duration(seconds))return NM_INVALID;
    if(!nm_ieee_f32())return NM_UNSUPPORTED;
    if(bytes<12||bytes>max_bytes||bytes>UINTPTR_MAX-(uintptr_t)p)return NM_BOUNDS;
    if(memcmp(p,"RIFF",4)||memcmp(p+8,"WAVE",4)||nm_u32(p+4)!=bytes-8)return NM_INVALID;
    size_t pos=12,offset=0,data_size=0;const uint8_t *format=NULL;int have_data=0;
    while(pos<bytes){
        if(bytes-pos<8)return NM_INVALID;
        const uint8_t *tag=p+pos;size_t count=nm_u32(tag+4);pos+=8;
        if(count>bytes-pos)return NM_INVALID;
        if(!memcmp(tag,"fmt ",4)){
            if(format||have_data||(count!=16&&count!=18))return NM_INVALID;
            format=p+pos;if(count==18&&nm_u16(format+16))return NM_UNSUPPORTED;
        }else if(!memcmp(tag,"data",4)){
            if(!format||have_data)return NM_INVALID;
            have_data=1;offset=pos;data_size=count;
        }
        pos+=count;
        if(count&1){if(pos>=bytes||p[pos])return NM_INVALID;pos++;}
    }
    if(!format||!have_data)return NM_INVALID;
    uint16_t code=nm_u16(format),channels=nm_u16(format+2),bits=nm_u16(format+14);
    if(channels!=1||nm_u32(format+4)!=16000)return NM_UNSUPPORTED;
    if(!((code==1&&bits==16)||(code==3&&bits==32)))return NM_UNSUPPORTED;
    size_t sample_bytes=bits/8;
    if(nm_u16(format+12)!=sample_bytes||nm_u32(format+8)!=16000*sample_bytes)return NM_INVALID;
    if(!data_size||data_size%sample_bytes)return NM_INVALID;
    size_t count=data_size/sample_bytes;
    if(count>=480000 || (double)count> (double)(size_t)(seconds*16000.0))return NM_BOUNDS;
    if(code==3)for(size_t i=0;i<count;i++){
        uint32_t value=nm_u32(p+offset+i*4);
        if((value&0x7fffffffu)>0x3f800000u)return NM_INVALID;
    }
    nm_wav_info parsed;memset(&parsed,0,sizeof parsed);
    parsed.sample_count=count;parsed.sample_rate=16000;parsed.channels=1;
    parsed.encoding=code==1?NM_WAV_PCM16:NM_WAV_FLOAT32;parsed.duration_seconds=(double)count/16000.0;
    *info=parsed;*data_offset=offset;return NM_OK;
}
nm_status nm_wav_inspect(const void *blob,size_t bytes,size_t max_bytes,double seconds,nm_wav_info *info){
    if(!blob||!nm_aligned_info(info))return NM_INVALID;
    if(nm_overlap(blob,bytes,info,sizeof(*info)))return NM_ALIAS;
    nm_wav_info parsed;size_t offset;nm_status result=nm_wav_parse(blob,bytes,max_bytes,seconds,&parsed,&offset);
    if(result==NM_OK)*info=parsed;
    return result;
}
nm_status nm_wav_decode(const void *blob,size_t bytes,size_t max_bytes,double seconds,
                         float *samples,size_t capacity,nm_wav_info *info){
    if(!nm_aligned_info(info)||!nm_aligned_float(samples))return NM_INVALID;
    nm_wav_info parsed;size_t offset;nm_status result=nm_wav_parse(blob,bytes,max_bytes,seconds,&parsed,&offset);
    if(result!=NM_OK)return result;
    if(capacity<parsed.sample_count||parsed.sample_count>SIZE_MAX/sizeof(float))return NM_BOUNDS;
    size_t out_bytes=parsed.sample_count*sizeof(float);
    if(nm_overlap(blob,bytes,samples,out_bytes)||nm_overlap(blob,bytes,info,sizeof(*info))||
       nm_overlap(samples,out_bytes,info,sizeof(*info)))return NM_ALIAS;
    const uint8_t *data=(const uint8_t *)blob+offset;
    for(size_t i=0;i<parsed.sample_count;i++){
        if(parsed.encoding==NM_WAV_PCM16){
            int32_t value=nm_u16(data+i*2);if(value&0x8000)value-=65536;
            samples[i]=(float)value*(1.f/32768.f);
        }else{uint32_t value=nm_u32(data+i*4);memcpy(samples+i,&value,4);}
    }
    *info=parsed;return NM_OK;
}

static nm_status nm_pixels(size_t w,size_t h,size_t *pixels){
    if(!w||!h||!pixels)return NM_INVALID;
    if(w>SIZE_MAX/h||w*h>NM_IMAGE_MAX_PIXELS)return NM_BOUNDS;
    *pixels=w*h;return NM_OK;
}
static nm_status nm_span(size_t w,size_t h,size_t channels,size_t stride,size_t *span){
    size_t pixels;nm_status result=nm_pixels(w,h,&pixels);if(result!=NM_OK)return result;
    if(!channels||w>SIZE_MAX/channels)return NM_BOUNDS;
    size_t row=w*channels;
    if(stride<row||(h-1)>(SIZE_MAX-row)/stride)return NM_BOUNDS;
    *span=(h-1)*stride+row;return NM_OK;
}
static nm_status nm_scaled(size_t w,size_t h,size_t channels,size_t *horizontal){
    if(channels!=1&&channels!=3)return NM_UNSUPPORTED;
    size_t pixels;nm_status result=nm_pixels(w,h,&pixels);if(result!=NM_OK)return result;
    if(pixels>NM_IMAGE_MAX_PIXELS/16||w>SIZE_MAX/4||h>SIZE_MAX/4||pixels>SIZE_MAX/4/channels)return NM_BOUNDS;
    *horizontal=pixels*4*channels;return NM_OK;
}

/* Pillow's 8bpc BICUBIC contract: a=-0.5, normalized clipped support,
 * 22-bit signed coefficients and rounded/clipped horizontal intermediate.
 * Reference: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
 * This is an independent bounded implementation, not image inference. */
enum {NM_FILTER_BITS=22,NM_FILTER_ONE=1<<22};
typedef struct {size_t first;int count;int32_t weights[5];} nm_filter;
static double nm_cubic(double distance){
    double x=distance<0?-distance:distance;
    if(x<1)return ((1.5*x-2.5)*x)*x+1;
    if(x<2)return (((x-5)*x+8)*x-4)*-.5;
    return 0;
}
static nm_filter nm_coefficients(size_t input,size_t output_index){
    nm_filter result;double center=((double)output_index+.5)*.25,sum=0,weights[5]={0};
    int64_t first=(int64_t)(center-1.5),end=(int64_t)(center+2.5);
    if(first<0)first=0;
    if((uint64_t)end>input)end=(int64_t)input;
    result.first=(size_t)first;result.count=(int)(end-first);
    for(int k=0;k<result.count;k++){weights[k]=nm_cubic((double)(first+k)-center+.5);sum+=weights[k];}
    for(int k=0;k<5;k++){
        double value=sum?weights[k]/sum:weights[k];
        result.weights[k]=(int32_t)(value*NM_FILTER_ONE+(value<0?-.5:.5));
    }
    return result;
}
static uint8_t nm_filter_byte(int64_t sum){
    if(sum<=0)return 0;
    int64_t value=sum/NM_FILTER_ONE;
    return value>=255?255:(uint8_t)value;
}
static void nm_horizontal(const uint8_t *src,size_t w,size_t h,size_t channels,size_t stride,uint8_t *tmp){
    size_t output_w=w*4;
    for(size_t x=0;x<output_w;x++){
        nm_filter filter=nm_coefficients(w,x);
        for(size_t y=0;y<h;y++)for(size_t c=0;c<channels;c++){
            int64_t sum=NM_FILTER_ONE/2;
            for(int k=0;k<filter.count;k++)sum+=(int64_t)src[y*stride+(filter.first+(size_t)k)*channels+c]*filter.weights[k];
            tmp[(y*output_w+x)*channels+c]=nm_filter_byte(sum);
        }
    }
}
static void nm_vertical_u8(const uint8_t *tmp,size_t w,size_t h,size_t channels,uint8_t *out,size_t stride){
    size_t ow=w*4,oh=h*4;
    for(size_t y=0;y<oh;y++){
        nm_filter filter=nm_coefficients(h,y);
        for(size_t x=0;x<ow;x++)for(size_t c=0;c<channels;c++){
            int64_t sum=NM_FILTER_ONE/2;
            for(int k=0;k<filter.count;k++)sum+=(int64_t)tmp[((filter.first+(size_t)k)*ow+x)*channels+c]*filter.weights[k];
            out[y*stride+x*channels+c]=nm_filter_byte(sum);
        }
    }
}
nm_status nm_bicubic4_workspace_bytes(size_t w,size_t h,size_t channels,size_t *bytes){
    if(!bytes)return NM_INVALID;
    size_t count;nm_status result=nm_scaled(w,h,channels,&count);
    if(result==NM_OK)*bytes=count;
    return result;
}
nm_status nm_bicubic4_u8(const uint8_t *src,size_t src_bytes,size_t w,size_t h,size_t channels,size_t src_stride,
                         uint8_t *out,size_t out_bytes,size_t out_stride,void *work,size_t work_bytes){
    if(!src||!out||!work)return NM_INVALID;
    size_t tmp_bytes,source_span,output_span;nm_status result=nm_scaled(w,h,channels,&tmp_bytes);if(result!=NM_OK)return result;
    result=nm_span(w,h,channels,src_stride,&source_span);if(result!=NM_OK)return result;
    result=nm_span(w*4,h*4,channels,out_stride,&output_span);if(result!=NM_OK)return result;
    if(src_bytes<source_span||out_bytes<output_span||work_bytes<tmp_bytes)return NM_BOUNDS;
    if(nm_overlap(src,source_span,out,output_span)||nm_overlap(src,source_span,work,tmp_bytes)||
       nm_overlap(out,output_span,work,tmp_bytes))return NM_ALIAS;
    nm_horizontal(src,w,h,channels,src_stride,(uint8_t *)work);
    nm_vertical_u8((const uint8_t *)work,w,h,channels,out,out_stride);return NM_OK;
}

nm_status nm_sr_workspace_bytes(size_t w,size_t h,size_t *bytes){
    if(!bytes||!w||!h)return NM_INVALID;
    if(w>1024||h>1024||w>NM_SR_MAX_TILE_PIXELS/h)return NM_BOUNDS;
    *bytes=w*h*15;return NM_OK;
}
nm_status nm_sr_prepare(const uint8_t *src,size_t src_bytes,size_t w,size_t h,size_t channels,size_t stride,
                         int64_t start_x,int64_t start_y,size_t tw,size_t th,float *low,size_t low_count,
                         float *high,size_t high_count,void *work,size_t work_bytes){
    if(!nm_ieee_f32())return NM_UNSUPPORTED;
    if(!src||!work||!nm_aligned_float(low)||!nm_aligned_float(high))return NM_INVALID;
    if(channels!=3&&channels!=4)return NM_UNSUPPORTED;
    size_t required,source_span;nm_status result=nm_sr_workspace_bytes(tw,th,&required);if(result!=NM_OK)return result;
    result=nm_span(w,h,channels,stride,&source_span);if(result!=NM_OK)return result;
    size_t pixels=tw*th,lp=pixels*3,hp=pixels*48;
    if(start_x>INT64_MAX-(int64_t)(tw-1)||start_y>INT64_MAX-(int64_t)(th-1)||
       src_bytes<source_span||low_count<lp||high_count<hp||work_bytes<required)return NM_BOUNDS;
    size_t lb=lp*sizeof(float),hb=hp*sizeof(float);
    if(nm_overlap(src,source_span,low,lb)||nm_overlap(src,source_span,high,hb)||nm_overlap(src,source_span,work,required)||
       nm_overlap(low,lb,high,hb)||nm_overlap(low,lb,work,required)||nm_overlap(high,hb,work,required))return NM_ALIAS;
    uint8_t *tile=(uint8_t *)work,*tmp=tile+pixels*3;
    for(size_t y=0;y<th;y++)for(size_t x=0;x<tw;x++){
        int64_t yy=start_y+(int64_t)y,xx=start_x+(int64_t)x;
        size_t sy=yy<0?0:(uint64_t)yy>=h?h-1:(size_t)yy;
        size_t sx=xx<0?0:(uint64_t)xx>=w?w-1:(size_t)xx;
        for(size_t c=0;c<3;c++)tile[(y*tw+x)*3+c]=src[sy*stride+sx*channels+c];
    }
    for(size_t p=0;p<pixels;p++)for(size_t c=0;c<3;c++)low[c*pixels+p]=(float)tile[p*3+(2-c)];
    nm_horizontal(tile,tw,th,3,tw*3,tmp);
    size_t ow=tw*4,oh=th*4,high_pixels=pixels*16;
    for(size_t y=0;y<oh;y++){
        nm_filter filter=nm_coefficients(th,y);
        for(size_t x=0;x<ow;x++)for(size_t c=0;c<3;c++){
            int64_t sum=NM_FILTER_ONE/2;
            for(int k=0;k<filter.count;k++)sum+=(int64_t)tmp[((filter.first+(size_t)k)*ow+x)*3+(2-c)]*filter.weights[k];
            high[c*high_pixels+y*ow+x]=(float)nm_filter_byte(sum);
        }
    }
    return NM_OK;
}
nm_status nm_bgr_output_rgb(const float *scores,size_t count,size_t w,size_t h,uint8_t *rgb,size_t rgb_bytes,size_t stride){
    if(!nm_ieee_f32())return NM_UNSUPPORTED;
    if(!nm_aligned_float(scores)||!rgb)return NM_INVALID;
    size_t pixels,span;nm_status result=nm_pixels(w,h,&pixels);if(result!=NM_OK)return result;
    result=nm_span(w,h,3,stride,&span);if(result!=NM_OK)return result;
    if(pixels>SIZE_MAX/3/sizeof(float)||count<pixels*3||rgb_bytes<span)return NM_BOUNDS;
    if(nm_overlap(scores,pixels*3*sizeof(float),rgb,span))return NM_ALIAS;
    for(size_t i=0;i<pixels*3;i++)if(!nm_finite_f32(scores[i]))return NM_INVALID;
    for(size_t y=0;y<h;y++)for(size_t x=0;x<w;x++)for(size_t c=0;c<3;c++){
        float value=scores[(2-c)*pixels+y*w+x];if(value<0)value=0;if(value>1)value=1;
        float scaled=value*255.f;rgb[y*stride+x*3+c]=(uint8_t)scaled;
    }
    return NM_OK;
}
