#include "tts_logger.h"
#include "stdio.h"

void tts_log(TTS_LOG_CAT_t cat, const char * logStr)
{
    printf("%s", logStr);
}

