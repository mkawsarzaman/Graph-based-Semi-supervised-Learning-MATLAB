function graph=ConstructGraph(X,graphoptions)



%%% for LGC parameter
alpha=0.99;
%%% for GTAM parameter
mu=0.01;

if strcmp(graphoptions.GraphType,'BMATCHING');
    W=Bmatching_Graph(X,graphoptions.K,'NO');
elseif strcmp(graphoptions.GraphType,'ADAPTIVEBMATCHING');
    P=Bmatching_Graph(X,graphoptions.K,'NO');
    
    options = [];
    options.Metric = graphoptions.Metric;
    options.NeighborMode = 'KNN';
    options.k = 0;
    options.WeightMode = 'HeatKernel';
    options.Adaptive=graphoptions.Adaptive;
    options.bSelfConnected=0;
    W = constructW(X,options);        
    W=W.*P;
    
elseif strcmp(graphoptions.GraphType,'LLE');  
    W=LLE_Graph(X,graphoptions.K,graphoptions.Metric);
elseif strcmp(graphoptions.GraphType,'KNN');  
    options = [];
    options.Metric = graphoptions.Metric;
    options.NeighborMode = 'KNN';
    options.k = graphoptions.K;
    options.WeightMode = 'HeatKernel';
    options.t = 0.1;
    options.Adaptive=graphoptions.Adaptive;
    options.bSelfConnected=0;
    W = constructW(X,options);
end
graph.W=W;

%%% for LGC
D=full(diag(sum(W)));
D1=D^(-0.5);
L=D1*W*D1;
I=eye(length(W),length(W));
IS=(I-alpha*L)^(-1);

graph.L=L;
graph.IS=IS;

P=(L/mu+I)^(-1);
A=P'*L*P+(P'-I)*(P-I);

graph.A=A;
graph.P=P;


