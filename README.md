
fig, axs = plt.subplots(1, 2, figsize=(16,9))
axs[0].set_title('Maximum Likelihood curve')
axs[ 1].set_title("Newton's optimization")

while np.linalg.norm(first_der) > TOL:
#     clear_output(wait=True)
    counter += 1
    Z = np.ravel(beta) @ np.transpose(X)
    Sigmoid = 1/ (1 + np.exp(-Z))
    X_axis = np.ravel(Vector_X['Hours (xi)'].values.reshape(1,20))
    Y_axis = np.ravel(Sigmoid)
    
    axs[0].plot(X_axis, Y_axis)
    axs[0].set_xlabel('Hours (xi)')
    axs[0].set_ylabel('Probability Score')
    display.clear_output(wait=True)
    display.display(pl.gcf())
    axs[1].plot(range(counter+1), cost_val)
    axs[1].set_xlabel('Iteration No.')
    axs[1].set_ylabel('Loss')
    time.sleep(1.0)
