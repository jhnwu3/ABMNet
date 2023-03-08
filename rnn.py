# get data

# train - test split

# initialize model

# train model

# initialize current state, curr, (so that might be a t0 vector that i'll provide) 
# predicted_t2, hidden = model_rnn.forward(x_t1, curr)
# loss(predicted, moments_truth_t1)
# loss.backward() 
# optimizer.step() 
# repeat step 
# predicted_t3, hidden = model_rnn.forward(x_t2, hidden)
# loss blah blah and then just repeat that for all time points in training
#


# evaluate model on test 

