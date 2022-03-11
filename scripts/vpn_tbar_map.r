# Find TBar map for each vpn type
# mhturner@stanford.edu

# For each body/neuron:
#   1) Find output synapse locations in central brain by VPN type
#   2) convert T-Bar locations from hemibrain raw space to desired JRC2018 space
#
# Saves:
#   outputs_by_region: (T-bar) count per Ito region (vpn type x ito region)
#   output_mask: T-Bar density image in JRC2018 space. Each vpn type is assigned a unique integer value, decoded by...
#   vpn_code
#
# # References:
#   http://natverse.org/neuprintr/
#   https://groups.google.com/forum/embed/?place=forum%2Fnat-user&pli=1#!topic/nat-user/zNaCyQnZeVg
#   http://natverse.org/nat.templatebrains/reference/xform_brain.html


library(nat.flybrains)
library(nat.jrcbrains)
library(nat.h5reg)
library(neuprintr)
library(dplyr)
library(bioimagetools)
library(rhdf5)

options(warn=1)

data_dir = '/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync/template_brain'
data_dir = '/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync/template_brain/all_vpns'

t0 = Sys.time()

# Load atlas
res = 0.38 # um/voxel of atlas
ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'ito_2018.tif'), as.is=TRUE)

# Get neuron / body IDs for different projection neuron types
VPNs = neuprint_search("type:(LC|LPLC|LLPC|LT|MC)[0-9].*")
# VPNs = neuprint_search("type:(LC|LPLC)[0-9].*")

vpn_types = unique(VPNs[,'type'])

# Init results matrices
output_density <- array(0, dim=dim(ito_atlas))
output_mask <- array(0, dim=dim(ito_atlas))
outputs_by_region <- matrix(0, length(vpn_types), max(ito_atlas))

all_body_ids = VPNs[,'bodyid']
# all_body_ids = sample_n(data.frame(all_body_ids), 350)[,1] # testing

# split into chunks for less gigantic neuprint calls
# chunks = split(all_body_ids, ceiling(seq_along(all_body_ids)/100)) # testing
chunks = split(all_body_ids, ceiling(seq_along(all_body_ids)/500))

for (c_ind in 1:length(chunks)){
  body_ids = neuprint_ids(chunks[[c_ind]])

  # get synapses associated with bodies
  syn_data = neuprint_get_synapses(body_ids)
  syn_data = syn_data[!duplicated(syn_data[2:5]),] # remove duplicate T-bar locations (single t-bar -> multiple postsynapses)

  print(sprintf('Loaded chunk %s: syn_data size = %s x %s', c_ind, dim(syn_data)[1],  dim(syn_data)[2]))

  # convert hemibrain raw locations to microns
  syn_data[,c("x", "y", "z")] = syn_data[,c("x", "y", "z")] * 8/1000 # vox -> um

  # Go from hemibrain space to atlas space
  syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JRC2018F) / res # x,y,z um -> atlas voxels

  # split into input / output
  input_synapses = as.data.frame(syn_data[syn_data$prepost==1, c("x", "y", "z", "bodyid")])
  output_synapses = as.data.frame(syn_data[syn_data$prepost==0, c("x", "y", "z", "bodyid")])

  # For each cell in synapse list
  ct = 0
  for (body_id in body_ids){
    cell_type = filter(VPNs, bodyid==body_id)['type']
    cell_type_code = which(vpn_types == cell_type[,,])

    if (length(cell_type_code) > 0){
      # Swap x and y for indexing
      input_yxz = data.matrix(input_synapses[input_synapses$bodyid==body_id, c("y", "x", "z")])
      output_yxz = data.matrix(output_synapses[output_synapses$bodyid==body_id, c("y", "x", "z")])

      mode(input_yxz) = 'integer' # floor to int to index
      mode(output_yxz) = 'integer' # floor to int to index

      # # # # # # ITO ATLAS # # # # # # # # # # # #
      # Get regions + counts for cell outputs
      output_regions = ito_atlas[output_yxz]
      output_regions = output_regions[output_regions!=0] # remove non-region locations
      output_tab = table(output_regions)
      output_regions = as.numeric(names(output_tab))  # now unique regions
      output_counts = as.vector(output_tab)
      if (length(output_regions) > 0){
        # Add new t-bars to outputs_by_region
        outputs_by_region[cell_type_code, output_regions] =
          outputs_by_region[cell_type_code, output_regions] +
          t(replicate(length(cell_type_code), output_counts))

        # Append output synapse counts to output_mask
        ct_by_vox = aggregate(data.frame(output_yxz)$x, by=data.frame(output_yxz), length)
        output_density[data.matrix(ct_by_vox)[,1:3]] = output_density[data.matrix(ct_by_vox)[,1:3]] + data.matrix(ct_by_vox)[,4]

        output_mask[data.matrix(ct_by_vox)[,1:3]] = cell_type_code
      }
      ct = ct + 1

    }
    else {
      print(sprintf('Skipped body_id %s; cell type = %s', body_id, cell_type))
      }

  } # end body_ids

  print(sprintf('Completed chunk %s: total cells = %s', c_ind, ct))

} # end chunks

# Save outputs_by_region, output_mask, and type key
write.csv(outputs_by_region, file.path(save_dir, 'vpn_all_outputs_by_region.csv'))
write.csv(data.frame(vpn_types), file.path(save_dir, 'vpn_all_types.csv'))

h5createFile(file.path(save_dir, 'vpn_all_glom_map.h5'))
h5createGroup(file.path(save_dir, 'vpn_all_glom_map.h5'), "density")
h5write(output_density, file.path(save_dir, 'vpn_all_glom_map.h5'), name="density/array", index=list(NULL,NULL,NULL))

h5createGroup(file.path(save_dir, 'vpn_all_glom_map.h5'), "mask")
h5write(output_mask, file.path(save_dir, 'vpn_all_glom_map.h5'), name="mask/array", index=list(NULL,NULL,NULL))

Sys.time() - t0
