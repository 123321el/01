# ############## 根据对txt文件 写入、读取数据，绘制曲线图##############
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import numpy as np
#
# X=[0,1,2,3,4,5,6,7,8,9]
# Y1 = [0.61, 0.41,0.10, 0.015,0.011,0.002, 0.001,0.002,0.003,0.002]
# # Y2 = [1.39, 1.13,0.71,0.955,0.87,0.535,0.586,0.485,0.647,0.281,0.130,0.128,0.50,0.27,0.28]
# # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)
#
# plt.figure(figsize=(8,6))  # 定义图的大小
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("V3 model loss change curve", )
#
# l1=plt.plot(X,Y1,color="green",linewidth=2)
# # l2=plt.plot(X,Y2,color="blue",linewidth=2)
# # l2=plt.plot(X,Y2)
# my_font=fm.FontProperties(fname=" C:\\Windows\\Fonts\\simkai.ttf")
#
#
# plt.show()

text="本发明的目的在于提供一种勘探能力强、适应复杂环境的海底作业机器人。本发明公开的海底作业机器人的技术方案是:一种海底作业机器人，包括主机架"
print(len(text))


