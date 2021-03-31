import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers



class CustomFit(keras.Model):
	def __init__(self,model):
		super(CustomFit,self).__init__()
		self.model = model


	def train_step(self,data):
		for batch_idx,(x,y) in enumerate(data):
			with tf.GradientTape() as tape:

				y_pred = self.model(x,training=True)
				loss = self.complied_loss(y,y_pred)


			training_vars = self.trainable_variables
			gradients = tape.gradient(loss,training_vars)

			self.optimizer.apply_gradient(zip(gradients,training_vars))

			self.complied_metrics.update_state(y,y_pred)

		return {m.name : m.result() for m in self.metrics}	




################################

# for epoch in range(epochs):
# 	print(f"\n start of Training Epoch {epoch}")
# 	for batch_idx,(x_batch,y_batch)in enumerate(ds_train):
# 		with tf.GradientTape() as tape:
# 			y_pred = model(x_batch,training=True)
# 			loss = loss_fn(y_batch,y_pred)

# 		gradients = tape.gradients(loss,model.trainable.weights)
# 		optimizer.apply_gradient(zip(gradients,model.trainable_weights))
# 		acc_metric.update_state(y_batch,y_pred)


# 	train_acc = acc_metric.result()
# 	print(f"Accuracy over Epoch {train_acc}")
# 	acc_metric.reset_states()



