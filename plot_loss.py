
import pickle
import numpy as np
import matplotlib.pyplot as plt


# train_errors_per_epoch_excluding_testset_2.txt format: epoch, average_train_loss, dev_loss, disp_error, final_disp_error

# test_results.pkl format: target and pred pair in a tuple: [(2m X T, 2m X T), (2m X T, 2m X T),...]

# loss_file_name = 'try_copy.txt'
loss_folder = 'log/'
# loss_file_name = loss_folder + 'before_dropout_0.01L2/train_errors_per_epoch_excluding_testset_[2].txt'
loss_file_name = loss_folder + 'train_errors_per_epoch_excluding_testset_[2].txt'



all_loss = np.loadtxt(loss_file_name, delimiter=',')
# print(all_loss)

# end_ep = 31
start = 0
end_ep = all_loss.shape[0] + 1

# end_ep = 500

ep = all_loss[start:end_ep,0]
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


ax1.plot(ep, all_loss[start:end_ep,1], 'r--', label='average train error')
ax1.plot(ep, all_loss[start:end_ep,2], 'b--', label='dev error')
ax2.plot(ep, all_loss[start:end_ep,3], 'g--', label='displacement error')
ax2.plot(ep, all_loss[start:end_ep,4], 'y--', label='final dispplacement error')

fig.suptitle("Error history in training")
ax1.grid(True)
ax2.grid(True)
ax1.legend(loc='upper center')
ax2.legend(loc='upper center')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
plt.show()

