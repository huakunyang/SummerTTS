#ifndef _TTS_PLT_LOG_H_
#define _TTS_PLT_LOG_H_

typedef enum
{
    TTS_LOG_ERROR=0,
    TTS_LOG_WARNING,
    TTS_LOG_INFO
}TTS_LOG_CAT_t;

void tts_log(TTS_LOG_CAT_t cat, const char * logStr);

#endif
