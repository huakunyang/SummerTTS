#include "utils.h"
#include "stdint.h"
#include "tts_logger.h"
#include "tts_file_io.h"
#include "stdlib.h"
#include "stdio.h"

int ttsLoadModel(char * ttsModelName, float **ttsModel)
{
    TTS_STAT_t st;
    if(-1 == tts_stat(ttsModelName, &st))
    {
        return -1 ;
    }

    TTS_FILE_t *fp = tts_fopen(ttsModelName);
    if(!fp)
    {
        tts_log(TTS_LOG_ERROR,"TTS_SYNC: Fail to open am model file\n");
        return -1;
    }

    //printf("file size :%d\n",st.size_);
    float * modelData = (float *)malloc(st.size_);

    tts_fread(modelData,st.size_,fp);
    tts_fclose(fp);

    *ttsModel = modelData;
    
    return st.size_;
}

void tts_free_data(void * data)
{
     free(data);
}
