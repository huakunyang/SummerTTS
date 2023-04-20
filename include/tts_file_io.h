#ifndef _TTS_PLT_FILE_IO_H_
#define _TTS_PLT_FILE_IO_H_

#include "stdint.h"

typedef struct
{
    int32_t size_;
}TTS_STAT_t;

typedef struct
{
    void *fp_;
}TTS_FILE_t;

int32_t tts_stat(char * filePath, TTS_STAT_t * ttsSt);
TTS_FILE_t * tts_fopen(char * filePath);
void tts_fclose(TTS_FILE_t * ttsFP);
int32_t tts_fread(void * buf,int32_t size, TTS_FILE_t * ttsFP);

#endif
