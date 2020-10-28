for j in range (hidden_size):
    list(rnn.hidden2hidden.parameters())[0].data[j, j].data.copy_(torch.tensor(0))
    for i in range (hidden_size):
        sign = 1
        if j >= hidden_size * 4 / 5:
            sign = -1
        if (list(rnn.hidden2hidden.parameters())[0].data[i, j].item() * sign < 0):
            list(rnn.hidden2hidden.parameters())[0].data[i, j].data.copy_(torch.tensor(0))

