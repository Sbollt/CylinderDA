%function cyl_navier_stokes
clear; close all

%% Create mesh using distmesh2d (open source)

% mesh shape (distances)
xlimR = 10; % length of pipe
pipeR = 3; % radius of pipe
pipeLead = 3; % distance from inlet to centre of cylinder
cylR = 0.5; % radius of cylinder

% create function to define shape of mesh:
%  - shape of rectangle
%  - position and shape of cylinder
fd=@(p) ddiff(drectangle(p,-pipeLead,xlimR,-pipeR,pipeR),...
    dcircle(p,0,0,cylR));

% create function to define mesh refinement
r0 = 0.4;
fh=@(p) 0.3+min(1*(sqrt(sum(p.^2,2))-r0),2.5);

% create mesh using distmesh2d (open source package for meshing)
% nodePosition: position of nodes
% triIndex: indices for the triangles
figh = figure;
[nodePosition,triIndex]=distmesh2d(fd,fh,0.11,...
    [-pipeLead,-pipeR;xlimR,pipeR],...
    [-pipeLead,-pipeR;-pipeLead,pipeR;xlimR,-pipeR;xlimR,pipeR]);

%% Create mesh and reference element
% get:
%  - basis functions (Legendre polyonimals)
%  - quadrature rule to evaluate integrals (Gauss quadrature)
% use Taylor-Hood polynomials:
%  - velocity: order p polynomials
%  - pressure: order p-1 polynomials
% meshes{1}/refs{1}: x-velocity
% meshes{2}/refs{2}: y-velocity
% meshes{3}/refs{3}: pressure

% get mesh
[meshes,refs,ops] = make_mesh_cyl(nodePosition,triIndex,...
    xlimR,pipeR,pipeLead,cylR);

%% solve FE problem

% set equation
dim = 2; eqn = CylFlow(dim); eqn.sym_adv = false; mu0 = 0;

% set options for newton solver
newtonopt = struct('verbosity',0,'maxiter',25);

% get degrees of freedom
ndof2 = size(meshes{1}.coord,1);
ndof1 = size(meshes{3}.coord,1);

% get inner product
ops.X = cyl_inner_product(eqn,meshes,refs,ops);

% get interior and boundary nodes
bnodes = nodesonbgrp(meshes{1},1:5);
inodes = setdiff((1:ndof2)',bnodes);

% get stokes flow solution
Ud = construct_dirichlet_constant_lift(eqn,meshes,refs,mu0);

% perturb the flow to create asymmetry
U1 = Ud; % stokes flow solution
Up = 0.1*sin(pi*meshes{1}.coord(:,2)/pipeR);
U1(inodes) = U1(inodes) + Up(inodes);

% set equation options
mmu{1} = 100; eqn.mu(1) = 100; % set Reynolds number, eqn.mu(1)=Re
eqn.steadyBool = true; % ignore this
resfun = @(U,t) compute_residual_vec(eqn,meshes,refs,ops,U,t);
Mtime = construct_mass(eqn,meshes,refs);

% time marching (finite element)
timeopt.order = 3; timeopt.h = 0.2; nt0 = 150;
timeopt.timedomain = [0,timeopt.h];
tt0 = (0:timeopt.h:(timeopt.h*(nt0-1)))';
UU0 = zeros(length(U1), nt0); UU0(:,1) = U1;
for i = 2:nt0
    if (i <= timeopt.order)
        U0 = UU0(:,1);
    else
        U0 = UU0(:,i-1:-1:i-timeopt.order);
    end
    [~,Ui,~] = bdf(resfun,Mtime,U0,timeopt,newtonopt);
    UU0(:, i) = Ui(:, end);
    
    if (false) % plot solutions (only if you want)
        figure(1), clf, % x-velocity
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{3},refs{3},UU0(2*ndof2 + (1:ndof1),i));
        axis equal; xlim([-pipeLead,xlimR]);ylim([-pipeR,pipeR]);
        colorbar;
        drawnow
        
        figure(2), clf, % pressure
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{1},refs{1},UU0(1:ndof2, i));
        axis equal; xlim([-pipeLead,xlimR]);ylim([-pipeR,pipeR]);
        colorbar;
        drawnow;
    end
end

% discard some data (ignore start-up, consider periodic solution)
iStart = 50;
UU = UU0(:,iStart:end); tt = tt0(iStart:end); nt = length(tt);

% compute kinetic energy and drag
pqt = timeopt.order;
opt_out.terminalBool = false; opt_out_T.terminalBool = true;
outfun_fe = @(U,t) compute_output_vec(eqn,meshes,refs,ops,U,opt_out,t);
outfun_T_fe = @(U,t) compute_output_vec(eqn,meshes,refs,ops,U,opt_out_T,t);

eqn.out_ke = true;
ke_fe = compute_output_unsteady(pqt,outfun_fe,outfun_T_fe,tt,UU);
eqn.out_ke = false;
drag_fe = compute_output_unsteady(pqt,outfun_fe,outfun_T_fe,tt,UU);

%% Get reduced-order model

% Get solution snapshots for POD
% subtract the Stokes solution so that POD snapshots have
% homogeneous Dirichlet boundary conditions
UU0pod = UU0-Ud; UUpod = UU-Ud;

% Get POD modes
Nmax = nt0; podtol = 1e-8;
Zmax = pod(ops.X,reshape(UU0pod,[ops.ndof,nt0]),Nmax,podtol);

% set options for reduced basis (POD) solver
rb_opt = struct('lifting','particular','mu0',mu0);

% choose N (number of basis functions)
% use error in Kinetic Energy to decide
errN = Inf; N = 0;
while errN > 1e-2
    N = N + 1;
    Z = Zmax(:,1:N);
    rb_ops = construct_rb(eqn,meshes,refs,Z,rb_opt);
    
    % get equations for RB problems in primal space
    outfun_rb = @(U,t) compute_eqp_output(eqn,rb_ops,U,opt_out,t);
    outfun_T_rb = @(U,t) compute_eqp_output(eqn,rb_ops,U,opt_out_T,t);
    
    % compute output
    aalpha_rb_p = Z'*ops.X*UUpod;
    eqn.out_ke = true;
    ke_rb_p = compute_output_unsteady(pqt,outfun_rb,outfun_T_rb,tt,aalpha_rb_p);
    
    % compute error
    errN = abs(ke_rb_p-ke_fe)/abs(ke_fe);
end

%% get coordinates for pressure sensors
% choose pressure sensors on surface of cylinder

% get nodes on surface of cylinder
cyl_nodes = nodesonbgrp(meshes{3},5);
cyl_coord = meshes{3}.coord(cyl_nodes,:);

% get coordinates for nodes we want to keep
% compute angle for each node, and organize by angle
thetaVec = sort(atan2(cyl_coord(:,2),cyl_coord(:,1)));
n_skip = 2; % number of nodes to skip
cyl_coord_sparse = cylR*[cos(thetaVec(1:n_skip:end)),...
    sin(thetaVec(1:n_skip:end-1))]; % compute cylinder coordinates

% get index for the cyl_coord_sparse (closest node to desired node)
cyl_nodes_sparse = zeros(length(cyl_coord_sparse),1);
for idx = 1:length(cyl_coord_sparse)
    thisDistance = vecnorm(cyl_coord-cyl_coord_sparse(idx,:),2,2);
    [~,thisIndex] = min(thisDistance);
    cyl_nodes_sparse(idx) = cyl_nodes(thisIndex);
end
cyl_nodes_sparse = unique(cyl_nodes_sparse);

% update cyl_coord_sparse
cyl_coord_sparse = meshes{3}.coord(cyl_nodes_sparse,:);

% convert cyl_nodes_sparse to global
cyl_nodes_global = cyl_nodes_sparse + 1 + 2*ndof2;

% plot nodes of mesh
figure(3);
plot(meshes{3}.coord(:,1),meshes{3}.coord(:,2),'o');
axis equal; xlim([-pipeLead,xlimR]);ylim([-pipeR,pipeR]); hold on;
plot(cyl_coord(:,1),cyl_coord(:,2),'o');
plot(cyl_coord_sparse(:,1),cyl_coord_sparse(:,2),'gx');
legend("All nodes","Cylinder nodes","Sparse cylinder nodes");

%% apply Ensemble Kalman filter to reduced-order model
% needs to be implemented

% get equations for RB problems in primal space
resfun_rb = @(U,t) compute_eqp_residual(eqn,rb_ops,U,t);
massfun_rb = @(U,t) compute_eqp_mass(eqn,rb_ops,U,t);

% get basis functions and lifting function to compute pressure
Z_sparse = Z(cyl_nodes_global,:);
Ud_sparse = Ud(cyl_nodes_global);

% solve RB problem, and apply ENKF
n_pop = 10;
aalpha_rb = zeros(N,nt, n_pop);
aalpha_rb(:,1,:) = repmat(Z'*ops.X*UUpod(:,1),[1,n_pop]); % project FE sol onto RB space

%TODO Intialize N realizations with noise
a0 = aalpha_rb(:,1);
Chat = zeros(N, N, nt);
mhat = zeros(N,nt);


%TODO Encapsulate
for i = 1:nt
    pressure_observed_i = zeros(length(cyl_nodes_global),n_pop);
    pressure_predicted_i = zeros(length(cyl_nodes_global),n_pop);
    for j = 1:n_pop
        % get solutions for time-marching
        if (i <= timeopt.order)
            a0 = aalpha_rb(:,1, j);
        else
            a0 = aalpha_rb(:,i-1:-1:i-timeopt.order,j);
        end
  
        %Add noise for first loop
        if i == 1
          a0 = a0 + randn(size(a0))*std(a0);
        end
        
        % apply time-marching
        %\hat{v}_{j+1}
        [~,ai,~] = bdf(resfun_rb,massfun_rb,a0,timeopt,newtonopt);
        
        % get pressure data (predicted by RB model)
        H = [Z_sparse] ;
        pressure_predicted_i(:,j) = H*ai(:,end);
        %pressure_predicted = Z_sparse*ai(:,end)+Ud_sparse;
        ai_enkf = ai(:,end);
  
  
        % Perturbed pressure
        % get pressure data (observed)
        %TODO Add perturbations
        pressure_observed_i(:,j) = UU(cyl_nodes_global,i) - Ud_sparse;
        
        
        % save RB solution (prediction)
        % These are \hat{v}_j+1 (for now)
        aalpha_rb(:, i, j) = ai_enkf;
    end
    % update solution using ENKF
    %Population mean
    mhat(:,i) = (1/n_pop)* sum(aalpha_rb(:,i,:),3);
    %Calcuate population covariance
    for j = 1:n_pop
      Chat(:,:,i) = Chat(:,:,i) + (aalpha_rb(:,i,j) - mhat(:,i))*(aalpha_rb(:,i,j) - mhat(:,i))';
    end
    Chat(:,:,i) = (1/(n_pop - 1))*Chat(:,:,i);

    % Calculate Kalman matricies for update
    S = H*Chat(:,:,i)*H.';
    K = Chat(:,:,i)*H.'*inv(S);

    %Update time steps (\hat{v}_{j+1} \mapsto v_{j+1}
    for j = 1:n_pop
      aalpha_rb(:,i,j) = (eye(length(K,1),length(H,2)) - K*H)*aalpha_rb(:,i,j) + K*pressure_observed_i(:,j);
    end

    %% Check error
    ensemble_pressure_error = norm(mean(pressure_observed_i - pressure_predicted_i,2))


end

% compute solutions
UU_rb = Z*aalpha_rb + Ud;
UU_rb_p = Z*aalpha_rb_p + Ud;

% compute output
eqn.out_ke = true;
ke_rb = compute_output_unsteady(pqt,outfun_rb,outfun_T_rb,tt,aalpha_rb);
eqn.out_ke = false;
drag_rb = compute_output_unsteady(pqt,outfun_fe,outfun_T_fe,tt,UU_rb);
drag_rb_p = compute_output_unsteady(pqt,outfun_fe,outfun_T_fe,tt,UU_rb_p);

% compute error in KE and drag
ke_err_rb = abs(ke_rb-ke_fe)/abs(ke_fe);
ke_err_rb_p = abs(ke_rb_p-ke_rb)/abs(ke_fe);
ke_err_fe_p = abs(ke_rb_p-ke_fe)/abs(ke_fe);
drag_err_rb = abs(drag_rb-drag_fe)/abs(drag_fe);
drag_err_rb_p = abs(drag_rb_p-drag_rb)/abs(drag_fe);
drag_err_fe_p = abs(drag_rb_p-drag_fe)/abs(drag_fe);
