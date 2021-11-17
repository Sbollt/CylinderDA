%function cyl_navier_stokes
clear; close all

% Getting the appropriate directories
addpath(genpath('../../'));

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

%% Get reduced-order model

% Get solution snapshots for POD
% subtract the Stokes solution so that POD snapshots have
% homogeneous Dirichlet boundary conditions
UU0pod = UU0-Ud; UUpod = UU-Ud;

% Get POD modes
Nmax = nt0; podtol = 1e-8;
Zmax = pod(ops.X,reshape(UU0pod,[ops.ndof,nt0]),Nmax,podtol);

% choose N (number of basis functions)
N = 4; Z = Zmax(:,1:N);
rb_opt = struct('lifting','particular','mu0',mu0);
rb_ops = construct_rb(eqn,meshes,refs,Z,rb_opt);

%% get coordinates for pressure sensors
% choose pressure sensors on surface of cylinder

% get nodes on surface of cylinder
cyl_nodes = nodesonbgrp(meshes{3},5);
cyl_coord = meshes{3}.coord(cyl_nodes,:);

% get coordinates for nodes we want to keep
% compute angle for each node, and organize by angle
thetaVec = sort(atan2(cyl_coord(:,2),cyl_coord(:,1)));
n_skip = 2;
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
np = length(cyl_nodes_sparse);

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
drawnow;

%% apply Ensemble Kalman filter to reduced-order model
% needs to be implemented

% get equations for RB problem
resfun_rb = @(U,t) compute_eqp_residual(eqn,rb_ops,U,t);
massfun_rb = @(U,t) compute_eqp_mass(eqn,rb_ops,U,t);

% get basis functions and lifting function to compute pressure
Z_sparse = Z(cyl_nodes_global,:);
Ud_sparse = Ud(cyl_nodes_global);

% solve RB problem (using time-marching)
a0 = Z'*ops.X*UUpod(:,1); % get initial RB coefficients using projection
timeopt_rb = timeopt; % copy time-marching structure
timeopt_rb.timedomain = [0,(nt-1)*timeopt_rb.h]; % update time domain
eqn.mu(1) = mmu{1}(1);
[~,aalpha_rb,~] = bdf(resfun_rb,massfun_rb,a0,timeopt_rb,newtonopt);

% solve RB problem, but apply ENKF
n_ens = 5;
aalpha_enkf_mat = zeros(N,nt,n_ens);
aalpha_enkf_mat(:,1,:) = repmat(aalpha_rb(:,1),[1,1,n_ens]);

%TODO Intialize N realizations with noise
Chat = zeros(N,N);
mhat = zeros(N,nt);
gammaO = 1e-3; % covariance for observations
sigmaV = 1e-2; % covariance for state vector
IC_perturb = 1e-2;
H = Z_sparse; % linear observor operator
sScale = 1; % s in equation (10.11), noise or noise in observations

%Add noise to initial conditions
a0 = aalpha_rb(:,1);
for jdx = 1:n_ens
    thisXi = randn(N,1)*IC_perturb;
    aalpha_enkf_mat(:,1,jdx) = a0+a0.*thisXi;
end

%TODO Encapsulate
for i = 1:nt
    % initialize matrices to hold pressure values
    pressure_observed_i = zeros(length(cyl_nodes_global),n_ens);
    pressure_predicted_i = zeros(length(cyl_nodes_global),n_ens);
    
    % prediction step
    for j = 1:n_ens
        % get solutions for time-marching
        if (i <= timeopt.order)
            a0 = aalpha_enkf_mat(:,1,j);
        else
            a0 = aalpha_enkf_mat(:,i-1:-1:i-timeopt.order,j);
        end
        
        % apply time-marching (update \hat{v}_{j+1})
        [~,ai,~] = bdf(resfun_rb,massfun_rb,a0,timeopt,newtonopt);
        ai_enkf = ai(:,end);
        
        % add noise to state vector
        thisXi = sigmaV*randn(N,1);
        ai_enkf = ai_enkf + ai_enkf.*thisXi;
        
        % apply observor operator (H\hat{v}_{j+1})
        pressure_predicted_i(:,j) = H*ai_enkf+Ud_sparse;
        
        % observe pressure (y_{j+1})
        thisEta = gammaO*randn(np,1); % noise
        thisP = UU(cyl_nodes_global,i); % pressure
        pressure_observed_i(:,j) = thisP + sScale*thisEta;
        
        % save RB solution (prediction)
        % These are \hat{v}_j+1 (for now)
        aalpha_enkf_mat(:,i,j) = ai_enkf;
    end
    %Population mean
    mhat(:,i) = (1/n_ens)* sum(aalpha_enkf_mat(:,i,:),3);
    %Calcuate population covariance
    Chat = zeros(N,N);
    for j = 1:n_ens
        thisDiff = aalpha_enkf_mat(:,i,j)-mhat(:,i);
        Chat = Chat + thisDiff*thisDiff';
    end
    Chat = Chat/(n_ens-1);
    
    % analysis step
    % Calculate Kalman matricies for update
    S = H*Chat*H'+gammaO*eye(np);
    K = Chat*H'/S;
    
    %Update time steps (\hat{v}_{j+1} \mapsto v_{j+1}
    for j = 1:n_ens
        aij_enkf = aalpha_enkf_mat(:,i,j);
        aalpha_enkf_mat(:,i,j) = aij_enkf - K*(H*aij_enkf+Ud_sparse) + K*pressure_observed_i(:,j);
    end
end

% get EnKF RB coefficients
aalpha_enkf = mhat; % take ensemble mean
aalpha_rb_p = Z'*ops.X*(UU-Ud); % get projected solution

% compute solutions in FE space
UU_enkf = Z*aalpha_enkf + Ud;
UU_rb_p = Z*aalpha_rb_p + Ud;
UU_rb = Z*aalpha_rb + Ud;

sol_diff = UU-UU_enkf;
sol_diff_p = UU-UU_rb_p;
sol_diff_rb = UU-UU_rb;

% compute error (FE)
err_mat = zeros(nt,3);
for tdx = 1:size(UU,2)
    err_mat(tdx,1) = sqrt(sol_diff(:,tdx)'*ops.X*sol_diff(:,tdx));
    err_mat(tdx,2) = sqrt(sol_diff_rb(:,tdx)'*ops.X*sol_diff_rb(:,tdx));
    err_mat(tdx,3) = sqrt(sol_diff_p(:,tdx)'*ops.X*sol_diff_p(:,tdx));
end

% normalize error (FE)
U_mean = mean(UU,2);
fe_norm = sqrt(U_mean'*ops.X*U_mean);
err_mat = err_mat/fe_norm;

% compute mean solution (for normalization purposes)
a_mean = mean(aalpha_rb_p,2);

% compute error (RB)
err_mat_rb = zeros(nt,2);
for tdx = 1:nt
    err_mat_rb(tdx,1) = norm(aalpha_enkf(:,tdx)-aalpha_rb_p(:,tdx))/norm(aalpha_rb_p(:,tdx));
    err_mat_rb(tdx,2) = norm(aalpha_rb(:,tdx)-aalpha_rb_p(:,tdx))/norm(aalpha_rb_p(:,tdx));
end

% compute RMS
rms_mat = zeros(nt,1);
for tdx = 1:nt
    thisRMS = 0;
    for qdx = 1:n_ens
        thisRMS = thisRMS + norm(aalpha_enkf_mat(:,tdx,qdx)-aalpha_enkf(:,tdx))^2/norm(aalpha_enkf(:,tdx))^2;
    end
    rms_mat(tdx) = sqrt(thisRMS/(n_ens-1));
end

% get time-step for plotting
tPlot = 0:timeopt.h:(timeopt.h*(nt-1));

% plot error (FE)
figure;
plot(tPlot,err_mat(:,1),'o',tPlot,err_mat(:,2),'o',tPlot,err_mat(:,3),'o');
xlabel("Time"); ylabel("Error");
legend('EnKF','RB',"RB (projected)",'location','best');

% plot error (RB)
figure;
plot(tPlot,err_mat_rb(:,1),'o',tPlot,err_mat_rb(:,2),'o');
xlabel("Time"); ylabel("Relative Error (%)");
legend('EnKF','RB','location','best');

% plot RMS
figure;
plot(tPlot,rms_mat,'o');
xlabel("Time"); ylabel("RMS");
