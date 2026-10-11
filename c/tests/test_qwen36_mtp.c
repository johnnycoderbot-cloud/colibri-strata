/* Pretrained-cell geometry/math, bounded ownership and causal cache oracle.
 * Embedded golden rows were computed independently with Torch F32/INT8
 * projections, Gemma norms, embedding-first FC, gated GQA and SiLU MLP.
 * --oracle-bin additionally consumes immutable real pretrained weights and
 * captured states with positions explicitly rebased to a synthetic prefix.
 * Neither fixture is a speed or full-prefix model-quality benchmark. */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#ifdef COLI_VULKAN
#undef COLI_VULKAN
#endif
#define COLI_STRATA_NATIVE 1
#define COLI_QWEN_MTP_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include <assert.h>

enum { MH=16, MF=24, MV=37, MS=6, MK=16 };
static const float oracle_norm_32[96]={
1.32523799f,-1.98116958f,0.56843996f,0.749892592f,1.24844015f,0.823713601f,-1.59045768f,0.212447301f,
-1.10751939f,0.219615728f,0.697022796f,-0.705463767f,0.634272218f,-1.26209939f,0.218389362f,-0.304869592f,
-1.49026954f,-0.568439543f,-0.276574522f,-0.308122694f,2.30809331f,-0.720277548f,0.0281203389f,-1.27804577f,
-0.33525303f,0.88702029f,-0.427649379f,0.568038225f,-0.797636688f,1.06456244f,-1.523857f,-0.639546752f,
-1.31893241f,-0.104556672f,-0.928535104f,1.49998045f,0.599733412f,-0.787060559f,-0.0684554055f,-1.42450392f,
0.506874144f,-0.266867161f,1.28354585f,0.764018357f,-1.00883985f,-0.105522893f,-1.61511743f,1.38366079f,
-0.241266385f,-0.512762249f,-0.0526107922f,1.5917455f,-1.48113215f,0.602052927f,-1.34763694f,-0.596968055f,
0.437331289f,0.683157086f,1.25302649f,-1.70847631f,0.19567807f,-0.23119989f,-0.776527226f,1.59306419f,
1.71191454f,-0.774768829f,0.649974465f,-1.12121439f,0.0709142685f,0.282019079f,0.139612764f,1.77446771f,
-1.03309476f,0.204075381f,-1.0319525f,-0.189220414f,0.282229602f,-0.596498489f,1.69635653f,-1.42394006f,
-0.0346450992f,1.2684691f,0.48123464f,-1.23778033f,-0.675743163f,-1.0473901f,0.513081133f,1.40383613f,
0.511579394f,0.813289344f,-1.17318201f,-0.305528075f,-0.961810112f,1.84683001f,1.01770961f,-0.822951853f
};
static const float oracle_logits_32[222]={
-1.57758451f,4.02945089f,-3.29073787f,-0.979026318f,0.940482318f,2.06630683f,0.747788489f,0.60611552f,
0.395713657f,-0.882367611f,0.549474001f,-1.57265043f,4.03438473f,-1.96680212f,-0.974092305f,0.94541651f,
2.07124066f,0.7527228f,0.611049533f,0.400647879f,-0.8774333f,0.554408252f,-1.05812228f,4.03931856f,
-1.96186769f,-0.969157875f,0.950350583f,2.07617497f,-3.83942628f,0.615983665f,0.405581892f,-0.872499347f,
0.559342504f,-1.05318785f,4.04425335f,-1.95693421f,-0.9642241f,-0.881762683f,0.581520736f,-2.47705364f,
3.00157356f,-0.819443345f,1.53724647f,1.66228914f,0.967906475f,1.43052077f,-4.53402472f,1.93449748f,
-0.813610435f,0.649673104f,-3.05066133f,3.06972623f,-0.751291335f,1.60539854f,1.73044145f,1.03605855f,
1.4986732f,-4.46587229f,2.00264931f,1.31277323f,0.717825055f,-2.98250914f,3.13787818f,-0.683139026f,
1.67355061f,0.47959289f,1.10421085f,1.56682503f,-4.39772034f,2.0708015f,1.38092542f,0.785977185f,
-2.91435671f,3.20603013f,-0.603955209f,0.680382371f,-4.42531443f,4.20742369f,-2.10439277f,2.18237519f,
-0.759295523f,2.11031199f,0.876716733f,0.466468513f,2.22170091f,-0.573070288f,0.711267173f,-6.5489912f,
4.23830938f,-2.07350779f,2.21326089f,-0.728410542f,2.14119697f,0.907601833f,0.497353733f,2.25258613f,
-1.16142082f,0.742152214f,-6.51810646f,4.26919365f,-2.0426228f,2.24414587f,-0.940137565f,2.17208195f,
0.938486755f,0.528238595f,2.28347135f,-1.13053572f,0.773037314f,-6.48722172f,4.30007839f,-1.19204891f,
1.93148792f,-3.67054796f,-0.410562634f,-2.32763386f,0.413685292f,-0.799701631f,3.43295979f,-1.0632422f,
3.72968125f,0.175797507f,-1.18054402f,1.94299328f,-3.7811203f,-0.399057508f,-2.31612802f,0.425190657f,
-0.788196325f,3.44446492f,-1.05173647f,3.74118662f,0.187302858f,0.416151106f,1.95449853f,-3.76961517f,
-0.387552023f,-2.30462313f,0.436695784f,-1.96649861f,3.45597053f,-1.04023135f,3.75269175f,0.198808193f,
0.427656412f,1.96600366f,-3.75810957f,-0.376046658f,0.555021226f,-1.08477211f,2.89827466f,-2.07881165f,
3.08286595f,-1.60850215f,0.89049679f,-1.50908327f,-0.903924108f,-1.64238715f,-1.56703746f,0.542577147f,
-1.09721637f,4.39402342f,-2.0912559f,3.07042217f,-1.62094629f,0.878053427f,-1.52152717f,-0.916368484f,
-1.65483093f,-1.57948136f,1.00366688f,-1.10966051f,4.3815794f,-2.1037004f,3.05797791f,-1.63339055f,
-0.932155788f,-1.53397167f,-0.928812742f,-1.66727519f,-1.59192586f,0.991222978f,-1.12210464f,4.3691349f,
-2.1161437f,0.908800185f,-3.71844053f,4.95276022f,-1.3174473f,0.0473388433f,-0.962487221f,-0.0264606029f,
-0.282901913f,-2.88450432f,0.931101203f,-3.90583158f,0.877790511f,-3.74945116f,6.0384016f,-1.3484571f,
0.0163292624f,-0.993496656f,-0.0574702248f,-0.313911498f,-2.91551399f,0.900091231f,-3.93684149f,2.7339282f,
-3.78045964f,6.00739193f,-1.37946689f,-0.0146805551f,-1.02450657f,2.85486078f,-0.34492141f,-2.94652414f,
0.869081914f,-3.96785116f,2.70291805f,-3.81147003f,5.97638226f,-1.41047645f
};
static const float oracle_norm_8[96]={
1.32187617f,-1.97937477f,0.570387483f,0.756138444f,1.24122655f,0.81932044f,-1.58964837f,0.21812728f,
-1.1102109f,0.223280326f,0.697093606f,-0.718676507f,0.633859754f,-1.26206219f,0.226858199f,-0.304694027f,
-1.49932039f,-0.564169645f,-0.276477635f,-0.31211406f,2.31117535f,-0.719457448f,0.0326403156f,-1.26647902f,
-0.329877764f,0.878816187f,-0.439572752f,0.577592373f,-0.802489877f,1.06457412f,-1.52039862f,-0.634774327f,
-1.31942832f,-0.107725509f,-0.928089321f,1.50170565f,0.598997295f,-0.785152733f,-0.0634412244f,-1.42057681f,
0.507963061f,-0.267753959f,1.28248417f,0.769609034f,-1.01204467f,-0.107092947f,-1.61775029f,1.37889647f,
-0.242038131f,-0.499668688f,-0.0562636033f,1.59430873f,-1.4775598f,0.595337451f,-1.34091794f,-0.608426571f,
0.437512428f,0.679402411f,1.25268555f,-1.71005344f,0.188703328f,-0.225590348f,-0.786090612f,1.59908378f,
1.71468019f,-0.773516238f,0.655791163f,-1.12048674f,0.0759139732f,0.279773116f,0.135563225f,1.77158678f,
-1.03850234f,0.205135703f,-1.02823043f,-0.193751156f,0.284155577f,-0.593344748f,1.70002306f,-1.41822088f,
-0.0329661109f,1.26598942f,0.486590296f,-1.24211919f,-0.676716149f,-1.04415512f,0.504683554f,1.40789926f,
0.513501227f,0.81705606f,-1.17596781f,-0.304392904f,-0.956304967f,1.84281099f,1.01882088f,-0.824947178f
};
static const float oracle_logits_8[222]={
-1.57334018f,4.02337456f,-3.29080987f,-0.995751381f,0.950484574f,2.06993675f,0.723412991f,0.619561315f,
0.392048806f,-0.862487018f,0.532325566f,-1.56835926f,4.02835417f,-1.96230888f,-0.990771115f,0.955465138f,
2.07491708f,0.728393734f,0.624541819f,0.3970294f,-0.857506275f,0.53730613f,-1.04528189f,4.03333569f,
-1.95732856f,-0.985790312f,0.960445762f,2.07989812f,-3.85954404f,0.629522502f,0.402009696f,-0.852525771f,
0.542286634f,-1.04030132f,4.0383153f,-1.95234776f,-0.980809808f,-0.844813049f,0.574668109f,-2.48899984f,
3.0099287f,-0.793028712f,1.53042853f,1.63434839f,0.956030071f,1.43203402f,-4.54605579f,1.92946887f,
-0.776845515f,0.642635763f,-3.06256747f,3.07789636f,-0.725061238f,1.59839618f,1.70231593f,1.02399743f,
1.50000179f,-4.4780879f,1.99743629f,1.33031702f,0.710603476f,-2.99459958f,3.14586401f,-0.657093465f,
1.66636395f,0.46119073f,1.09196532f,1.56796944f,-4.41012001f,2.06540346f,1.39828444f,0.778571188f,
-2.92663145f,3.21383166f,-0.59366107f,0.668146074f,-4.4335084f,4.22566557f,-2.096385f,2.16595244f,
-0.753786981f,2.1158216f,0.875852227f,0.451993942f,2.23224378f,-0.562799037f,0.699008107f,-6.5561738f,
4.25652838f,-2.06552315f,2.19681501f,-0.722924829f,2.14668369f,0.906714499f,0.482856154f,2.26310587f,
-1.15322995f,0.729870319f,-6.52531195f,4.28739023f,-2.03466082f,2.22767735f,-0.942027986f,2.17754602f,
0.937576532f,0.513718069f,2.2939682f,-1.1223681f,0.760732532f,-6.49444962f,4.31825209f,-1.19246483f,
1.93719983f,-3.67343187f,-0.421129227f,-2.32990694f,0.433542192f,-0.812214434f,3.42461514f,-1.04036486f,
3.73981118f,0.150621533f,-1.18082261f,1.94884217f,-3.79234338f,-0.409487069f,-2.3182652f,0.445184529f,
-0.800572336f,3.4362576f,-1.02872264f,3.7514534f,0.162263677f,0.407296926f,1.96048415f,-3.78070092f,
-0.397844821f,-2.30662251f,0.456826627f,-1.94835556f,3.44789982f,-1.01708031f,3.76309586f,0.173906088f,
0.418939352f,1.97212648f,-3.76905847f,-0.386202574f,0.538325429f,-1.06293404f,2.89517164f,-2.10162854f,
3.08191824f,-1.57825232f,0.879179835f,-1.51818669f,-0.894987106f,-1.62053847f,-1.58888996f,0.525576651f,
-1.07568288f,4.40411282f,-2.11437726f,3.06916904f,-1.59100115f,0.866430938f,-1.53093565f,-0.907736421f,
-1.63328731f,-1.60163867f,0.988822043f,-1.0884316f,4.39136457f,-2.12712646f,3.05642009f,-1.60375035f,
-0.941175878f,-1.5436846f,-0.920485198f,-1.64603615f,-1.61438799f,0.976073265f,-1.10118055f,4.37861538f,
-2.13987494f,0.904682279f,-3.71520352f,4.95477581f,-1.31784427f,0.0393290557f,-0.966189206f,-0.020702444f,
-0.285365433f,-2.90137434f,0.930224419f,-3.89407611f,0.873618603f,-3.74626708f,6.05279112f,-1.34890795f,
0.00826553348f,-0.997252882f,-0.0517660081f,-0.31642887f,-2.93243861f,0.899160564f,-3.92513967f,2.73844218f,
-3.77733088f,6.02172661f,-1.37997186f,-0.0227983147f,-1.02831638f,2.85475707f,-0.347492874f,-2.96350193f,
0.868097007f,-3.95620346f,2.70737839f,-3.80839467f,5.99066257f,-1.41103542f
};

static void close_enough(const float *a,const float *b,size_t n,float atol,float rtol) {
    for(size_t i=0;i<n;i++) {
        if(!q36_mtp_finite(a+i,1)||fabsf(a[i]-b[i])>atol+rtol*fabsf(b[i])) {
            fprintf(stderr,"oracle mismatch at %zu: %.9g vs %.9g (abs %.9g)\n",i,a[i],b[i],fabsf(a[i]-b[i]));abort();
        }
    }
}
static void model_fixture(Model *m) {
    memset(m,0,sizeof(*m));m->c.hidden=MH;m->c.inter=MF;m->c.vocab=MV;
    m->c.q_heads=4;m->c.kv_heads=2;m->c.head_dim=8;m->c.rotary_dim=4;
    m->c.eps=1e-6f;m->c.theta=10000.f;m->c.partial_rotary_factor=.5f;
    m->embed=malloc(MV*MH*4);float *lm=malloc(MV*MH*4);assert(m->embed&&lm);
    m->lm_head=(QW){.w=lm,.I=MH,.O=MV};
    for(int i=0;i<MV*MH;i++){m->embed[i]=(float)((i*37+13)%251-125)/99.f;lm[i]=(float)((i*19+71)%239-119)/103.f;}
}
static void make_weights(Model *m,Q36MtpWeights *w,float *owned[15]) {
    int rows[15],cols[15];q36_mtp_dims(&m->c,m->c.inter,rows,cols);
    for(int t=0;t<15;t++) {
        w->counts[t]=(size_t)rows[t]*cols[t];owned[t]=malloc(w->counts[t]*4);assert(owned[t]);w->values[t]=owned[t];
        for(size_t i=0;i<w->counts[t];i++)owned[t][i]=(float)((int)((i*(19+3*t)+31*(t+1))%251)-125)/(cols[t]==1?4000.f:700.f);
    }
}
static void run_fixture(void) {
    Model target;model_fixture(&target);Q36MtpWeights weights;float *owned[15];make_weights(&target,&weights,owned);
    float hidden[MS*MH],out[MH+2],logit[MV+2],k[MK],v[MK],ks[MS*MK],vs[MS*MK];
    int ids[MS]={2,7,13,19,23,31};
    for(int i=0;i<MS*MH;i++)hidden[i]=(float)((i*23+7)%211-105)/33.f;
    for(int bits=8;bits<=32;bits+=24) {
        Q36Mtp *m=q36_mtp_from_weights(&target,&weights,MS,bits);assert(m&&q36_mtp_capacity(m)==MS);
        assert(m->qwidth==2*MH && m->shadow.c.q_head_dim==16);
        for(int p=0;p<MS;p++) {
            out[0]=out[MH+1]=89.f;logit[0]=logit[MV+1]=91.f;
            assert(q36_mtp_predict_one(m,hidden+p*MH,ids[p],p,out+1,logit+1));
            close_enough(out+1,(bits==32?oracle_norm_32:oracle_norm_8)+p*MH,MH,8e-5f,8e-5f);
            close_enough(logit+1,(bits==32?oracle_logits_32:oracle_logits_8)+p*MV,MV,4e-4f,1e-4f);
            assert(out[0]==89.f&&out[MH+1]==89.f&&logit[0]==91.f&&logit[MV+1]==91.f);
            assert(q36_mtp_test_peek_allocated(m,p,ks+p*MK,vs+p*MK));
        }
        assert(m->full_cells==MS && m->lm_projections==MS && m->kv_projections==0);
        assert(!q36_mtp_truncate(m,MS+1));assert(!q36_mtp_truncate(m,-1));
        float *oldk=m->keys,*oldv=m->values;assert(!q36_mtp_reserve(m,INT_MAX));assert(m->keys==oldk&&m->values==oldv);
        assert(q36_mtp_reserve(m,MS+9)&&q36_mtp_length(m)==MS);
        for(int p=0;p<MS;p++) {
            assert(q36_mtp_test_peek_allocated(m,p,k,v));assert(!memcmp(k,ks+p*MK,MK*4));assert(!memcmp(v,vs+p*MK,MK*4));
        }
        assert(q36_mtp_truncate(m,0));assert(q36_mtp_kv_rows(m,hidden,ids,MS,0));
        assert(m->lm_projections==MS && m->full_cells==MS && m->kv_projections==MS);
        for(int p=0;p<MS;p++) {
            assert(q36_mtp_test_peek_allocated(m,p,k,v));assert(!memcmp(k,ks+p*MK,MK*4));assert(!memcmp(v,vs+p*MK,MK*4));
        }
        assert(q36_mtp_truncate(m,MS-1));assert(q36_mtp_predict_one(m,hidden+(MS-1)*MH,ids[MS-1],MS-1,out+1,logit+1));
        close_enough(out+1,(bits==32?oracle_norm_32:oracle_norm_8)+(MS-1)*MH,MH,8e-5f,8e-5f);
        /* Rejected operations leave the valid prefix and caller outputs intact. */
        assert(q36_mtp_truncate(m,0));for(int i=1;i<=MH;i++)out[i]=123.f;for(int i=1;i<=MV;i++)logit[i]=124.f;
        assert(!q36_mtp_predict_one(m,hidden,ids[0],1,out+1,logit+1));
        assert(!q36_mtp_predict_one(m,hidden,-1,0,out+1,logit+1));
        assert(!q36_mtp_predict_one(m,hidden,MV,0,out+1,logit+1));
        assert(!q36_mtp_predict_one(m,hidden,ids[0],0,hidden,logit+1));
        assert(!q36_mtp_predict_one(m,hidden,ids[0],0,out+1,out+1));
        assert(!q36_mtp_predict_one(m,hidden,ids[0],0,target.embed,logit+1));
        float save=hidden[3];hidden[3]=NAN;assert(!q36_mtp_kv_rows(m,hidden,ids,MS,0));hidden[3]=save;
        ids[MS-1]=-1;assert(!q36_mtp_kv_rows(m,hidden,ids,MS,0));ids[MS-1]=31;
        assert(!q36_mtp_kv_rows(m,hidden,ids,0,0));assert(!q36_mtp_kv_rows(m,hidden,ids,33,0));
        save=target.embed[ids[0]*MH];target.embed[ids[0]*MH]=INFINITY;
        assert(!q36_mtp_predict_one(m,hidden,ids[0],0,out+1,logit+1));target.embed[ids[0]*MH]=save;
        save=((float*)target.lm_head.w)[0];((float*)target.lm_head.w)[0]=NAN;
        assert(!q36_mtp_predict_one(m,hidden,ids[0],0,out+1,logit+1));((float*)target.lm_head.w)[0]=save;
        assert(q36_mtp_length(m)==0);
        for(int i=1;i<=MH;i++)assert(out[i]==123.f);for(int i=1;i<=MV;i++)assert(logit[i]==124.f);
        /* Source arrays may be freed: the cell owns immutable copied weights. */
        assert(q36_mtp_predict_one(m,hidden,ids[0],0,out+1,logit+1));q36_mtp_free(m);
    }
    assert(!q36_mtp_from_weights(&target,&weights,1,16));assert(!q36_mtp_from_weights(&target,&weights,0,8));
    weights.counts[14]--;assert(!q36_mtp_from_weights(&target,&weights,1,8));weights.counts[14]++;
    owned[14][0]=NAN;assert(!q36_mtp_from_weights(&target,&weights,1,8));owned[14][0]=0;
    for(int i=0;i<15;i++)free(owned[i]);free(target.embed);qw_free(&target.lm_head);
}

static void put_file(const char *dir,const char *name,const void *data,size_t n) {
    char path[1024];assert(snprintf(path,sizeof(path),"%s/%s",dir,name)>0);
    FILE *f=fopen(path,"wb");assert(f&&fwrite(data,1,n,f)==n&&fclose(f)==0);
}
static void put_u64(FILE *f,uint64_t value) {
    unsigned char p[8];for(int i=0;i<8;i++)p[i]=(unsigned char)(value>>(8*i));assert(fwrite(p,1,8,f)==8);
}
static void save_tensor_fixture(const char *dir,const Model *m,const Q36MtpWeights *w,int broken) {
    int rows[15],cols[15];q36_mtp_dims(&m->c,m->c.inter,rows,cols);
    char header[65536],path[1024];size_t used=0,offset=0;
    used+=(size_t)snprintf(header+used,sizeof(header)-used,"{");
    for(int i=0;i<15;i++) {
        const char *dtype=broken==1&&i==0?"F32\\u0000junk":broken==2&&i==0?"F16":"F32";
        size_t start=broken==3&&i==1?0:offset;
        used+=(size_t)snprintf(header+used,sizeof(header)-used,"%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d",i?",":"",q36_mtp_tensor_names[i],dtype,rows[i]);
        if(cols[i]!=1)used+=(size_t)snprintf(header+used,sizeof(header)-used,",%d",cols[i]);
        used+=(size_t)snprintf(header+used,sizeof(header)-used,"],\"data_offsets\":[%zu,%zu]}",start,start+w->counts[i]*4);
        offset+=w->counts[i]*4;assert(used<sizeof(header)-256);
    }
    used+=(size_t)snprintf(header+used,sizeof(header)-used,"}");
    assert(snprintf(path,sizeof(path),"%s/model-mtp.safetensors",dir)>0);FILE *f=fopen(path,"wb");assert(f);
    put_u64(f,used);assert(fwrite(header,1,used,f)==used);
    for(int i=0;i<15;i++)for(size_t j=0;j<w->counts[i];j++) {
        uint32_t u;memcpy(&u,w->values[i]+j,4);if(broken==4&&i==14&&j==0)u=0x7fc00000u;
        unsigned char p[4];for(int b=0;b<4;b++)p[b]=(unsigned char)(u>>(8*b));assert(fwrite(p,1,4,f)==4);
    }
    if(broken==5)assert(fputc(0,f)==0);assert(fclose(f)==0);
}
static void save_meta_fixture(const char *dir,const Model *m,const char *fp,int broken) {
    size_t n;unsigned char *data=q36_mtp_read(dir,"model-mtp.safetensors",Q36_MTP_WEIGHT_LIMIT,&n);assert(data);
    char sha[65];evidence_sha256_hex(data,n,sha);free(data);
    char text[65536];int rows[15],cols[15];q36_mtp_dims(&m->c,m->c.inter,rows,cols);
    size_t used=(size_t)snprintf(text,sizeof(text),
        "{\"schema\":\"colibri.qwen-mtp/v1\",\"tensor_dtype\":\"F32\","
        "\"norm_convention\":\"%s\",\"target_hidden_tap\":\"post_final_norm\","
        "\"shared_embedding\":true,\"shared_lm_head\":true,\"fc_order\":[\"%s\",\"%s\"],"
        "\"geometry\":{\"hidden_size\":%d,\"intermediate_size\":%d,\"num_attention_heads\":%d,"
        "\"num_key_value_heads\":%d,\"head_dim\":%d,\"vocab_size\":%d,\"num_hidden_layers\":%d,"
        "\"mtp_num_hidden_layers\":1,\"rotary_dim\":%d,\"rope_theta\":%.9g,\"rms_norm_eps\":%.9g,"
        "\"partial_rotary_factor\":%.9g,\"attn_output_gate\":true},"
        "\"native\":{\"fingerprint\":\"%s\",\"runtime_precision\":\"%s\"},"
        "\"sidecar\":{\"file\":\"model-mtp.safetensors\",\"bytes\":%zu,\"sha256\":\"%s\"},\"tensors\":{",
        broken==1?"full_gamma":"qwen_rmsnorm_plus_one",broken==2?"target_hidden":"embedding",broken==2?"embedding":"target_hidden",
        m->c.hidden+(broken==3),m->c.inter,m->c.q_heads,m->c.kv_heads,m->c.head_dim,m->c.vocab,m->c.n_layers,
        m->c.rotary_dim,m->c.theta,m->c.eps,m->c.partial_rotary_factor,fp,
        broken==4?"wrong precision":Q36_MTP_PRECISION,n,broken==5?"0000000000000000000000000000000000000000000000000000000000000000":sha);
    for(int i=0;i<15;i++) {
        used+=(size_t)snprintf(text+used,sizeof(text)-used,"%s\"%s\":{\"dtype\":\"F32\",\"source_dtype\":\"BF16\",\"shape\":[%d",i?",":"",q36_mtp_tensor_names[i],rows[i]);
        if(cols[i]!=1)used+=(size_t)snprintf(text+used,sizeof(text)-used,",%d",cols[i]);
        used+=(size_t)snprintf(text+used,sizeof(text)-used,"]}");assert(used<sizeof(text)-256);
    }
    used+=(size_t)snprintf(text+used,sizeof(text)-used,"}%s}",broken==6?",\"shared_embedding\":true":"");
    put_file(dir,"qwen36_mtp.json",text,used);
}
static void loader_fixture(void) {
    char dir[512];
#ifdef _WIN32
    char temp[MAX_PATH];assert(GetTempPathA(MAX_PATH,temp));int made=0;
    for(int i=0;i<100&&!made;i++) {
        assert(snprintf(dir,sizeof(dir),"%scoli-mtp-%lu-%d",temp,(unsigned long)GetCurrentProcessId(),i)>0);
        made=CreateDirectoryA(dir,NULL)!=0;
    }assert(made);
#else
    strcpy(dir,"/tmp/coli-mtp-XXXXXX");assert(mkdtemp(dir));
#endif
    Model m;model_fixture(&m);Q36MtpWeights w;float *owned[15];make_weights(&m,&w,owned);
    put_file(dir,"config.json","{}",2);put_file(dir,"tokenizer.json","{}",2);put_file(dir,"base.safetensors","identity",8);
    uint8_t digest[32];assert(sn_fingerprint_model(dir,Q36_MTP_PRECISION,digest)==SN_OK);
    char fp[65];const char hex[]="0123456789abcdef";
    for(int i=0;i<32;i++){fp[2*i]=hex[digest[i]>>4];fp[2*i+1]=hex[digest[i]&15];}fp[64]=0;
    /* Sidecar must be separate: remove it during target fingerprinting. */
    char side[1024];assert(snprintf(side,sizeof(side),"%s/side",dir)>0);
#ifdef _WIN32
    assert(CreateDirectoryA(side,NULL));
#else
    assert(mkdir(side,0700)==0);
#endif
    save_tensor_fixture(side,&m,&w,0);save_meta_fixture(side,&m,fp,0);
    Q36Mtp *cell=q36_mtp_init(&m,dir,side,MS);assert(cell&&cell->bits==8);q36_mtp_free(cell);
    for(int broken=1;broken<=6;broken++) {
        save_meta_fixture(side,&m,fp,broken);assert(!q36_mtp_init(&m,dir,side,MS));
    }
    for(int broken=1;broken<=5;broken++) {
        save_tensor_fixture(side,&m,&w,broken);save_meta_fixture(side,&m,fp,0);assert(!q36_mtp_init(&m,dir,side,MS));
    }
    save_tensor_fixture(side,&m,&w,0);save_meta_fixture(side,&m,fp,0);
    put_file(dir,"base.safetensors","changed target",14);assert(!q36_mtp_init(&m,dir,side,MS));
    put_file(dir,"base.safetensors","identity",8);
    /* Correct digest before an undigested data mutation still refuses. */
    char path[2048];assert(snprintf(path,sizeof(path),"%s/model-mtp.safetensors",side)>0);
    FILE *file=fopen(path,"r+b");assert(file&&fseek(file,-1,SEEK_END)==0);int last=fgetc(file);assert(last!=EOF);
    assert(fseek(file,-1,SEEK_END)==0&&fputc(last^1,file)!=EOF&&fclose(file)==0);assert(!q36_mtp_init(&m,dir,side,MS));
    assert(remove(path)==0);
#ifndef _WIN32
    assert(mkfifo(path,0600)==0);assert(!q36_mtp_init(&m,dir,side,MS));assert(remove(path)==0);
#endif
    assert(snprintf(path,sizeof(path),"%s/qwen36_mtp.json",side)>0);assert(remove(path)==0);
#ifdef _WIN32
    assert(RemoveDirectoryA(side));
#else
    assert(rmdir(side)==0);
#endif
    const char *names[]={"config.json","tokenizer.json","base.safetensors"};
    for(int i=0;i<3;i++){assert(snprintf(path,sizeof(path),"%s/%s",dir,names[i])>0);assert(remove(path)==0);}
#ifdef _WIN32
    assert(RemoveDirectoryA(dir));
#else
    assert(rmdir(dir)==0);
#endif
    for(int i=0;i<15;i++)free(owned[i]);free(m.embed);qw_free(&m.lm_head);
}
static void actual_oracle(const char *path) {
    FILE *f=fopen(path,"rb");assert(f);unsigned char header[48];assert(fread(header,1,48,f)==48&&!memcmp(header,"QMTO",4));
    assert(q36_mtp_u32(header+4)==1);Model t={0};
    t.c.hidden=(int)q36_mtp_u32(header+8);t.c.q_heads=(int)q36_mtp_u32(header+12);
    t.c.kv_heads=(int)q36_mtp_u32(header+16);t.c.head_dim=(int)q36_mtp_u32(header+20);
    t.c.inter=(int)q36_mtp_u32(header+24);t.c.vocab=(int)q36_mtp_u32(header+28);
    int S=(int)q36_mtp_u32(header+32);t.c.rotary_dim=(int)q36_mtp_u32(header+36);
    uint32_t u=q36_mtp_u32(header+40);memcpy(&t.c.eps,&u,4);u=q36_mtp_u32(header+44);memcpy(&t.c.theta,&u,4);
    assert(t.c.hidden==1024&&t.c.inter==3584&&S==8&&t.c.vocab==37);
    int r[15],c[15];q36_mtp_dims(&t.c,t.c.inter,r,c);Q36MtpWeights w;float *owned[15];
    for(int i=0;i<15;i++){w.counts[i]=(size_t)r[i]*c[i];owned[i]=malloc(w.counts[i]*4);assert(owned[i]);w.values[i]=owned[i];assert(fread(owned[i],4,w.counts[i],f)==w.counts[i]);}
    int D=t.c.hidden,V=t.c.vocab,K=t.c.kv_heads*t.c.head_dim,ids[8];
    t.embed=calloc((size_t)V*D,4);float *lm=malloc((size_t)V*D*4),*h=malloc((size_t)S*D*4);
    assert(t.embed&&lm&&h);t.lm_head=(QW){.w=lm,.I=D,.O=V};
    assert(fread(t.embed,4,(size_t)S*D,f)==(size_t)S*D);assert(fread(lm,4,(size_t)V*D,f)==(size_t)V*D);assert(fread(h,4,(size_t)S*D,f)==(size_t)S*D);
    for(int s=0;s<S;s++)ids[s]=s;
    float *want=malloc((size_t)S*(D+V+2*K)*4),*out=malloc((size_t)(D+V+2*K)*4);assert(want&&out);
    for(int mode=0;mode<2;mode++) {
        int bits=mode?8:32;assert(fread(want,4,(size_t)S*(D+V+2*K),f)==(size_t)S*(D+V+2*K));
        Q36Mtp *m=q36_mtp_from_weights(&t,&w,S,bits);assert(m);
        float max_hidden=0,max_logit=0,max_kv=0;
        for(int s=0;s<S;s++) {
            assert(q36_mtp_predict_one(m,h+(size_t)s*D,ids[s],s,out,out+D));
            assert(q36_mtp_test_peek_allocated(m,s,out+D+V,out+D+V+K));
            const float *gold[]={want+(size_t)s*D,want+(size_t)S*D+s*V,want+(size_t)S*(D+V)+s*K,want+(size_t)S*(D+V+K)+s*K};
            int counts[]={D,V,K,K};float *observed[]={out,out+D,out+D+V,out+D+V+K};
            for(int group=0;group<4;group++) {
                close_enough(observed[group],gold[group],(size_t)counts[group],8e-5f,8e-5f);
                for(int i=0;i<counts[group];i++){float e=fabsf(observed[group][i]-gold[group][i]);float *mx=group==0?&max_hidden:group==1?&max_logit:&max_kv;if(e>*mx)*mx=e;}
            }
        }
        printf("actual pretrained MTP source bits%d oracle: max hidden %.7g logits %.7g KV %.7g\n",bits,max_hidden,max_logit,max_kv);
        q36_mtp_free(m);
    }
    assert(fgetc(f)==EOF);fclose(f);for(int i=0;i<15;i++)free(owned[i]);free(t.embed);qw_free(&t.lm_head);free(h);free(want);free(out);
}
int main(int argc,char **argv) {
    run_fixture();loader_fixture();if(argc==3&&!strcmp(argv[1],"--oracle-bin"))actual_oracle(argv[2]);else assert(argc==1);
    puts("MTP cell: independent Torch oracle, native8/F32, KV-only, reserve, causal state and failure publication passed");return 0;
}
