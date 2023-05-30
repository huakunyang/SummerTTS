#ifndef _TTS_SYNTHESIZER_H_
#define _TTS_SYNTHESIZER_H_

#include "stdint.h"
#include "string"

using namespace std;

class SynthesizerTrn
{
public:
    SynthesizerTrn(float * modelData, int32_t modelSize);
    int16_t * infer(const string & line, int32_t sid, float lengthScale, int32_t & dataLen);
    int32_t getSpeakerNum();
    ~SynthesizerTrn();

private:
    void * priv_;
};

#endif
