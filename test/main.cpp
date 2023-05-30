#include "stdio.h"
#include "SynthesizerTrn.h"
#include "utils.h"
#include "string.h"


void convertAudioToWavBuf(
    char * toBuf, 
    char * fromBuf,
    int totalAudioLen)
{
    char * header = toBuf;
    int byteRate = 16 * 16000 * 1 / 8;
    int totalDataLen = totalAudioLen + 36;
    int channels = 1;
    int  longSampleRate = 16000;

    header[0] = 'R'; // RIFF/WAVE header
    header[1] = 'I';
    header[2] = 'F';
    header[3] = 'F';
    header[4] = (char) (totalDataLen & 0xff);
    header[5] = (char) ((totalDataLen >> 8) & 0xff);
    header[6] = (char) ((totalDataLen >> 16) & 0xff);
    header[7] = (char) ((totalDataLen >> 24) & 0xff);
    header[8] = 'W';
    header[9] = 'A';
    header[10] = 'V';
    header[11] = 'E';
    header[12] = 'f'; // 'fmt ' chunk
    header[13] = 'm';
    header[14] = 't';
    header[15] = ' ';
    header[16] = 16; // 4 bytes: size of 'fmt ' chunk
    header[17] = 0;
    header[18] = 0;
    header[19] = 0;
    header[20] = 1; // format = 1
    header[21] = 0;
    header[22] = (char) channels;
    header[23] = 0;
    header[24] = (char) (longSampleRate & 0xff);
    header[25] = (char) ((longSampleRate >> 8) & 0xff);
    header[26] = (char) ((longSampleRate >> 16) & 0xff);
    header[27] = (char) ((longSampleRate >> 24) & 0xff);
    header[28] = (char) (byteRate & 0xff);
    header[29] = (char) ((byteRate >> 8) & 0xff);
    header[30] = (char) ((byteRate >> 16) & 0xff);
    header[31] = (char) ((byteRate >> 24) & 0xff);
    header[32] = (char) (1 * 16 / 8); // block align
    header[33] = 0;
    header[34] = 16; // bits per sample
    header[35] = 0;
    header[36] = 'd';
    header[37] = 'a';
    header[38] = 't';
    header[39] = 'a';
    header[40] = (char) (totalAudioLen & 0xff);
    header[41] = (char) ((totalAudioLen >> 8) & 0xff);
    header[42] = (char) ((totalAudioLen >> 16) & 0xff);
    header[43] = (char) ((totalAudioLen >> 24) & 0xff);

    memcpy(toBuf+44, fromBuf, totalAudioLen);

}

#include "string"
#include "Hanz2Piny.h"
#include "hanzi2phoneid.h"
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char ** argv)
{
    const Hanz2Piny hanz2piny;
    const std::string file_path(argv[1]);

    if (!hanz2piny.isUtf8File(file_path)) 
    {
        printf("Failed to open UTF-8 encoding file: %s\n",file_path.c_str());
        return 0;
    }

    ifstream utf8_ifs(file_path);
    std::string line;
    getline(utf8_ifs, line);
    if (hanz2piny.isStartWithBom(line)) 
    {
        line = std::string(line.cbegin() + 3, line.cend());
    }
    

    float * dataW = NULL;
    int32_t modelSize = ttsLoadModel(argv[2],&dataW);

    SynthesizerTrn * synthesizer = new SynthesizerTrn(dataW, modelSize);

    int32_t spkNum = synthesizer->getSpeakerNum();
    
    printf("Available speakers in the model are %d\n",spkNum);

    if(spkNum > 20)
    {
        for(int spkID = 10; spkID <20; spkID ++)
        {
            int32_t retLen = 0;
            int16_t * wavData = synthesizer->infer(line,spkID, 1.1,retLen);

            char * dataForFile = new char[retLen*sizeof(int16_t)+44];
            convertAudioToWavBuf(dataForFile, (char *)wavData, retLen*sizeof(int16_t));

            char fileName[200] = {0};
            sprintf(fileName,"%s_%d.wav",argv[3],spkID);
        
            FILE * fpOut = fopen(fileName,"wb");
            fwrite(dataForFile, retLen*sizeof(int16_t)+44, 1, fpOut);
            fclose(fpOut);

            printf("%s generated\n",fileName);
            delete dataForFile;
            tts_free_data(wavData);
        }
    }
    else
    {
        int32_t retLen = 0;
        int16_t * wavData = synthesizer->infer(line,0, 1.0,retLen);

        char * dataForFile = new char[retLen*sizeof(int16_t)+44];
        convertAudioToWavBuf(dataForFile, (char *)wavData, retLen*sizeof(int16_t));

        FILE * fpOut = fopen(argv[3],"wb");
        fwrite(dataForFile, retLen*sizeof(int16_t)+44, 1, fpOut);
        fclose(fpOut);
        tts_free_data(wavData);
    }
        
    delete synthesizer;
    tts_free_data(dataW);

    return 0;
}
