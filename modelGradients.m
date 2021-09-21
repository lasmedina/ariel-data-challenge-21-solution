function [gradients,state,loss] = modelGradients(dlnet,dlX1,dlX2,Y)

[dlYPred,state] = forward(dlnet,dlX1,dlX2);

loss = mse(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

end
