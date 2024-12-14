# SummerTTS 用于纪念2023年即将到来和终将逝去的夏天

# 说明
- SummerTTS 是一个独立编译的语音合成程序(TTS)。可以本地运行不需要网络，而且没有额外的依赖，一键编译完成即可用于中文和英文的语音合成。
- SummerTTS 的底层计算库使用Eigen，Eigen是一套模板定义的函数，大部分情况下，只需要包含头文件即可，所以本项目没有其他依赖，在C++环境下可以独立编译和运行。
- 本项目使用Eigen提供的矩阵库实现了神经网络的算子，不需要依赖例如pytorch，tensorflow, ncnn 等其他NN运行环境。
- 本项目在 Ubuntu 上编译运行通过，其他类Linux平台，如Android，树莓派等，也应该没啥大问题，在Window上没有测试过，可能需要少许改动。
- 本项目的模型基于语音合成算法 vits, 在其基础上进行了基于C++的工程化
- 本项目适用MIT License，基于本项目的开发，使用人员或机构，请遵循 MIT License: https://mit-license.org

# 更新日志
- 2024-12-14: 添加License 信息为 MIT License: https://mit-license.org
- 2023-06-16: 更新添加一个更快的英文语音合成模型:single_speaker_english_fast.bin, 还是在如下网盘中，速度要快一些，合成的音质下降不明显:  
  链接: https://pan.baidu.com/s/1rYhtznOYQH7m8g-xZ_2VVQ?pwd=2d5h 提取码: 2d5h
- 2023-06-15: 支持纯英文的语音合成，需要同步最新的代码，使用下列网盘中的模型文件: single_speaker_english.bin, 以下面的方式合成英文语音：  
  ./tts_test ../test_eng.txt ../models/single_speaker_english.bin out_eng.wav  
  网盘路径如下，之前的中文语音合成和用法不受影响，需要说明的是本次更新只支持纯英文的语音合成，中文混合英文的暂时不支持。  
  链接: https://pan.baidu.com/s/1rYhtznOYQH7m8g-xZ_2VVQ?pwd=2d5h 提取码: 2d5h
- 2023-06-09: 新增了一个中等大小的单说话人模型: single_speaker_mid.bin  ，速度比之前的模型稍慢，但合成的音质似乎要好点（本人耳朵不算敏感，感觉要好点，也许是心理作用：P ），代码不需要更新，只需要在之前的网盘中下载 single_speaker_mid.bin 并使用即可. 
- 2023-06-08: 修改test/main.cpp, 支持换行和整篇文本的合成
- 2023-06-03: Fix 了昨天的版本中的一个错误，感谢热心网友Telen提供测试和线索，只有代码更新，模型不需要更新。 
- 2023-06-02: 大幅度提升了多音字发音合成的准确性，需要在百度网盘中获取新的模型，才能使用改善后的多音字发音和文本正则化（Text Normalization），今天更新的代码不能使用之前的模型，否则可能导致crash
- 2023-05-30: 集成 WeTextProcessing 作为前端文本正则化（Text Normalization）模块，极大的改善了对数字，货币，温度，日期等的正确发音合成。需要在下面的百度网盘中获取新的模型
- 2023-5-23： 使用新的算法大幅度提升了单说话人的语音合成速度。
- 2023-4-21： 初始创建


# 使用说明
- 将本项目的代码克隆到本地，最好是Ubuntu Linux 环境
- 从以下的百度网盘地址下载模型，放入本项目的model目录中：
  链接: https://pan.baidu.com/s/1rYhtznOYQH7m8g-xZ_2VVQ?pwd=2d5h 提取码: 2d5h
    
  模型文件放入后，models目录结构如下：    
  models/  
  ├── multi_speakers.bin  
  ├── single_speaker_mid.bin  
  ├── single_speaker_english.bin  
  ├── single_speaker_english_fast.bin  
  └── single_speaker_fast.bin  
  

- 进入Build 目录，执行以下命令：  
  cmake ..  
  make  
- 编译完成后，会在Build 目录中生成 tts_test 执行程序  
- 运行下列命令，测试中文语音合成（TTS）：  
  ./tts_test ../test.txt ../models/single_speaker_fast.bin out.wav   
- 运行下列命令，测试英文语音合成（TTS）：  
  ./tts_test ../test_eng.txt ../models/single_speaker_english.bin out_eng.wav  

  该命令行中：  
  第一个参数为是文本文件的路径，该文件包含需要被合成语音的文本。  
  第二个参数是前面提到的模型的路径，文件名开头的single 和 multi 表示模型包含了单个说话人还是多个说话人。推荐单说话人模型：single_speaker_fast.bin, 合成的速度较快，合成的音质也还行。
  第三个参数是合成的音频文件，程序运行完之后生成该文件，可以用播放器打开。
    
- 以上的测试程序实现在 test/main.cpp 中，具体合成的接口定义在 include/SynthesizerTrn.h， 如下：  
  int16_t * infer(const string & line, int32_t sid, float lengthScale, int32_t & dataLen)  

  该接口的：  
  第一个参数是待合成的语音的字符串。  
  第二个参数指定说话人的id 用于合成语音，该参数对多说话人模型有效，对单说话人模型，固定为0。说话人的个数可由接口 int32_t getSpeakerNum() 返回，有效id 为 0 到 该接口返回的说话人数量减1。  
  第三个参数 lengthScale 表示合成语音的语速，其值越大表示语速越慢。  
- 待合成的文本中可以包含阿拉伯数字和标点，但因为本项目的 文本正则化(TN) 模块还很粗糙，对于英文字符，会直接忽略。也因为文本正则化(TN) 模块还很粗糙，对不同语境下的多音字发音有时候会不准确。

# 后续开发
- 后续将开放模型训练和转化脚本
- 后续将尝试训练和提供音质更好的模型

# 联系作者
- 有进一步的问题或需要可以发邮件到 120365182@qq.com , 或添加微信: hwang_2011, 本人尽量回复。
  
# License
- 本项目适用MIT License，基于本项目的开发，使用人员或机构，请遵循 MIT License: https://mit-license.org

# 感谢
本项目在源代码和算法方面使用了下列方案，在此表示感谢, 若可能引发任何法律问题，请及时联系我协调解决
- Eigen  
- vits (https://github.com/jaywalnut310/vits)  
- vits_chinese (https://github.com/UEhQZXI/vits_chinese)
- MB-iSTFT-VITS (https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- WeTextProcessing (https://github.com/wenet-e2e/WeTextProcessing)
- glog (https://github.com/google/glog)
- gflags (https://github.com/gflags/gflags)
- openfst (https://github.com/kkm000/openfst)
- 汉字转拼音（https://github.com/yangyangwithgnu/hanz2piny）
- cppjieba (https://github.com/yanyiwu/cppjieba)
- g2p_en(https://github.com/Kyubyong/g2p)
- English-to-IPA(https://github.com/mphilli/English-to-IPA)
- 本项目的中文单说话人模型基于开源标贝数据集训练，多说话人模型基于开源数据集 aishell3 训练，英文单说话人模型基于LJ Speech 数据集。  




