for lr in [ 0.01, 0.001, 0.0001]:
    for wd in [0, 0.01, 0.001, 0.0001, 0.00001]:
        for mm in [0, 0.2, 0.5, 0.725, 0.9, 0.99]:
            for ld in [1, 0.95, 0.9]:
                hyperparameters = {'lr': lr, 'weight_decay': wd, 'momentum': mm}
                lr_decay = ld
                script = f'lr = {lr}, wd = {wd}, mm = {mm}, ld = {ld}'
                # save script to a csv file named 'script.csv'
                with open('script.csv', 'a') as f:
                    f.write(script + '\n')


                # plot the loss
                import matplotlib.pyplot as plt
                x = [i[0] for i in loss_values]
                y = [i[1] for i in loss_values]
                plt.plot(x, y)
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
