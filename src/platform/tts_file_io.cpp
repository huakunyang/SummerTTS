#include "tts_file_io.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include "stdio.h"
#include <cstring>


int32_t tts_stat(char * filePath, TTS_STAT_t * ttsSt)
{
    int32_t ret = -1;
    struct stat st;
    ret = stat(filePath, &st);
    if(ret != -1)
    {
        ttsSt->size_ = st.st_size;
    }
    
    return ret;
}

TTS_FILE_t * tts_fopen(char * filePath)
{
    TTS_FILE_t * ret = new TTS_FILE_t();
    memset(ret,0,sizeof(TTS_FILE_t));
    ret->fp_ = (void *)fopen(filePath,"rb");
    return ret;
}

void tts_fclose(TTS_FILE_t * ttsFP)
{
    if(NULL != ttsFP->fp_)
    {
        fclose((FILE *)ttsFP->fp_);
    }

    delete ttsFP;
}

int32_t tts_fread(void * buf,int32_t size, TTS_FILE_t * ttsFP)
{
    return fread(buf, size, 1, (FILE *)ttsFP->fp_);
}


