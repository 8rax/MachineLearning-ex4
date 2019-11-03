function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%transformer y pour passer du chiffre a un vecteur avec des 0 et des 1

%retourne la identity matrix sur 10*10
I = eye(num_labels);

%retourne une matrice avec que des 0 en 5000*10
Y = zeros(m, num_labels);

%retourne une matrice 5000*10 avec un 1 dans la colonne du chiffre y(i)
for i=1:m
  Y(i, :)= I(y(i), :);
end


%Propagation en avant

%Calcul du layer 1
%On rajoute l
X = [ones(m, 1) X];
layer1=X;
%On calcul le layer 2
layer2=sigmoid(X*Theta1');
% on rajoute le bias
layer2 = [ones(m, 1) layer2];
%On calcul le layer 3
layer3=sigmoid(layer2*Theta2');

%On calcul la cost function, avec la somme sur 1->m et la somme sur 1->K
J = sum(sum((-Y).*log(layer3) - (1-Y).*log(1-layer3), 2))/m;

%regularized cost function
%On calcul la pénalité, la somme de tout les theta au carré  sauf la premiere colonne
p = sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2));
%p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

%on calcul la cost function régularisée
J = sum(sum((-Y).*log(layer3) - (1-Y).*log(1-layer3), 2))/m + lambda*p/(2*m);
% -------------------------------------------------------------

% =========================================================================

%On calcul les sigma et on part de la derniere layer en faisant la diff entre ce qui est calculé et le vrai résultat 

sigma3=layer3.-Y;

% On propage au layer2 en utilisant le sigma3 et les theta du second layer et on multiplie par le sigmoid prime 
% en rajoutant une premiere colonne avec des 1 
sigma2=sigma3*Theta2.*sigmoidGradient([ones(size(X*Theta1', 1), 1) X*Theta1']);
% on garde que ce qu'il y a apres la premiere colonne
sigma2 = sigma2(:, 2:end);

% On accumule les gradient en multiplian sigma2 par le training set
delta_1 = (sigma2'*X);
% et le delta 2 pour sigma 3 * le layer2
delta_2 = (sigma3'*layer2);

Theta1_grad =delta_1./m;
Theta2_grad = delta_2./m;

%Pour regulariser on va d'abord calculer le deuxieme parametre a ajouter 
% en mettant a nul la premiere colonne de Theta1 et theta2

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];





end
