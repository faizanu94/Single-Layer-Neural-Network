function oneLayerNN_SSD
% one layer only, i.e. the hidden and the output
% loss is sum of squared differences, the beginner's classic, and sigmoid
% linlinearity
% full batch learning, no SGD right now


% load input dataset
examples = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% this particular input is n x m or features x examples 

nClasses = length(unique(labels));
nExamples = size(labels, 1);
nFeatures = size(examples, 1);

targets = zeros(10, size(labels, 1)); % row - number of classes, col - n examples - input was like that
% labels had class IDs
% so building a one-hot repn:
for n = 1 : size(labels, 1)
    targets(labels(n) + 1, n) = 1; 
end;

learningRate = 0.25; % fixed for now

nEpochs = 1000; % fixed for now

% add bias units
examples = horzcat(ones(nFeatures,1), examples);

%  setting up the synapses:
% the weight matrix is of size out x hidden, so counterintuitive in our fine
% tradition from line 1 of this code

% interval to sample weights from
a = -0.5;
b = 0.5;
weightsHidden2Output = (b-a) .* rand(nClasses, nFeatures) + a;
% % let's not saturate the sigmoid just yet - spread as much as the input
% % dims allow - not done
% weightsHidden2Output = weightsHidden2Output/ size(weightsHidden2Output, 2);

net = fitnet([nFeatures,nFeatures]);

% loop over epochs
for epochn = 1 : nEpochs
    epochn; % disp
    % loop over the entire set of examples for a single weight update
    for examplen = 1 : nExamples
    % forward propagation:
%        outputlayerOutput = sigmoidNonlinearity( weightsHidden2Output * examples(:, examplen) );
        outputlayerOutput(:, examplen) = 1/(1+exp(-(weightsHidden2Output*examples(:, examplen))));
        
    %     backward propagation:
    % compute error at output lauer
        delE = sum( targets(:,examplen) - outputlayerOutput(:,examplen )); % dE/dy
 %       delW = learningRate * (outputlayerOutput(:,examplen ) .* (1 - outputlayerOutput(:,examplen )))  * examples(:, examplen)'; % y(1-y) * x
        delW = learningRate * (outputlayerOutput(:,examplen ) .* (1 - outputlayerOutput(:,examplen )))  * examples(:, examplen)'; % y(1-y) * x
  %      delW = delW * delE;
        delW = delW * delE;
    end % for nExamples
%     weight update for batch = dataset
    error(epochn,:) = delE;
    accuracy(epochn,:) = 1 - delE;
    weightsHidden2Output = weightsHidden2Output + delW;
    
end % epochn
end