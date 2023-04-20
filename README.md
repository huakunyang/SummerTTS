# SummerTTS 用于纪念2023年即将到来和终将逝去的夏天

# 说明
- SummerTTS 是一个独立编译的语音合成程序(TTS)。
- SummerTTS 的底层计算库使用Eigen，Eigen是一套模板定义的函数，大部分情况下，只需要包含头文件即可，所以本项目基本上没有其他依赖，在C++环境下可以独立编译和运行。
- 本项目使用Eigen提供的矩阵库实现了神经网络的算子，不需要依赖其他NN运行环境，例如pytorch，tensorflow 等。
- 本项目在 Ubuntu 上编译运行通过，其他类Linux平台，如Android，树莓派等，也应该没啥大问题，在Window上没有测试过，可能需要少许改动。
- 本项目的模型基于语音合成算法 vits, 在其基础上进行了基于C++的工程化

# 使用说明
- 将本项目的代码克隆到本地，最好是Ubuntu Linux 环境
- 从以下的百度网盘地址下载模型，放入本项目的model目录中：

  链接: https://pan.baidu.com/s/15YivI-HfopuOfx3evZUj-g?pwd=vzjb 提取码: vzjb     
    
  models/  
  ├── multi_speaker_big.bin  
  ├── multi_speaker_small.bin  
  ├── multi_speker_medium.bin  
  ├── single_speaker_big.bin  
  ├── single_speaker_medium.bin  
  └── single_speaker_small.bin  

- 进入Build 目录，执行以下命令：  
  cmake ..  
  make  
- 编译完成后，会在Build 目录中生成 tts_test 执行程序  
- 运行下列命令，测试语音合成（TTS）：  
  ./tts_test ../test.txt ../models/single_speaker_small.bin out.wav   

  该命令行中：  
  第一个参数为是文本文件的路径，改文件包含需要被合成语音的文本。
    
  第二个参数是之前下载的模型的路径，文件名开头的single 和 multi 表示模型包含了单个说话人还是多个说话人。文件名结尾的big， medium， small 等分别表示模型的大小，模型越大，需要的计算量和内存越多，合成需要的时间越长，但合成的语音的质量效果越好。
    
  第三个参数是合成的音频文件，可以用播放器打开。
    
- 以上的测试程序实现在 test/main.cpp 中，具体合成的接口定义在 include/SynthesizerTrn.h， 如下：  
  int16_t * infer(const string & line, int32_t sid, float lengthScale, int32_t & dataLen)  

  该接口的：  
  第一个参数是待合成的语音的字符串。  
  第二个参数指定说话人的id 用于合成语音，该参数对多说话人模型有效，对单说话人模型，固定为0。说话人的个数可由接口 int32_t getSpeakerNum() 返回，有效id 为 0 到 该接口返回的说话人数量减1。  
  第三个参数 lengthScale 表示合成语音的语速，越大表示语速越慢。  
- 待合成的文本中可以包含阿拉伯数字和标点，但因为本项目的 文本正则化(TN) 模块还很粗糙，仅仅是将阿拉伯数字简单的念出，并将标点符号表示为停顿，对于英文字符，会直接忽略。也因为文本正则化(TN) 模块还很粗糙，对不同语境下的多音字发音有时候会不准确。

# 后续开发
- 后续将开放模型训练和转化脚本
- 后续也会对文本正则化(TN) 模块加强，使其更好的处理具体的数字的含义，例如日期，温度等。

# 联系作者
- 有进一步的问题或需要可以发邮件到 120365182@qq.com , 或添加微信: hwang_2011, 本人尽量回复。

# 感谢
本项目在源代码和算法方面使用了下列方案，在此表示感谢, 若可能引发任何法律问题，请及时联系我协调解决
- Eigen  
- vits (https://github.com/jaywalnut310/vits)  
- vits_chinese (https://github.com/UEhQZXI/vits_chinese)  
- 本项目的单说话人模型基于开源标贝数据集训练，多说话人模型基于开源数据集 aishell3 训练  




