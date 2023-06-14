# import matplotlib.pyplot as plt
# import numpy as np
#
# fig, ax = plt.subplots()
#
# labels = ['AAA','BBB','CCC','DDD','EEE','FFF','GGG','HHH','III','JJJ']
# data1 = [7, 17, 4, 9, 14, 6, 14, 16, 12, 9]
#
# indexes = np.argsort(-np.array(data1))
# data1_sorted = [data1[i] for i in indexes]
# labels_sorted = [labels[i] for i in indexes]
#
# y_pos = np.arange(len(labels))
#
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))
#
# ax.barh(y_pos, data1_sorted, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(labels_sorted)
# ax.invert_yaxis()  # labels read top-to-bottom
#
# plt.show()

import numpy as np
print(np.sqrt(6*np.sum(1/np.arange(1,1000000,dtype=np.float)**2)))
