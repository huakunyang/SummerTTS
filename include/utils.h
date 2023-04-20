#ifndef _TTS_UTILS_H_
#define _TTS_UTILS_H_

int ttsLoadModel(char * ttsModelName, float **ttsModel);
void tts_free_data(void * data);

#endif
