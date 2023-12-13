local module = {}

local base64 = require("base64")
local inspect = require("inspect")
local msgpack = require("MessagePack")
-- IPS seems to use SLB (https://code.google.com/archive/p/slb/)
local slb = require("SLB")

local function _pack(data)
  return base64.encode(msgpack.pack(data))
end

local function _unpack(string)
  return msgpack.unpack(base64.decode(string))
end

local function _type(obj)
  -- Returns the type of `obj` as a string.
  -- Unlike `type`, this function also works for IPS userdata.
  local luaType = type(obj)
  if luaType == "userdata" then
    -- IPS objects are userdata without a metatable. Try to look up
    -- using SLB but fall back to "userdata"
    return slb.type(obj) or "userdata"
  else
    return luaType
  end
end

local function vectorToTable(vector)
  local table = {}
  local vecType = _type(vector)
  local size
  if vecType == "Vector3d" or vecType == "Vector3i" then
    -- Vector3[id] does not have a size() method
    size = 3
  else
    size = vector:size()
  end

  for i = 1, size do
    -- IPS vectors are 0-indexed
    local element = vector[i - 1]

    -- If the element is a userdata, assume it's a vector for now
    if type(element) == "userdata" then
      table[i] = vectorToTable(element)
    else
      table[i] = element
    end
  end

  return table
end

local function range(from, to)
  -- Returns an array with values from `from` to `to`, inclusive.
  local arr = {}
  for v = from, to do
    arr[#arr + 1] = v
  end
  return arr
end

local function loadAndFitScene(scenePath)
  print("Loading scene " .. scenePath)
  local loaded = Ips.loadScene(scenePath)
  -- Fitting the scene helps with two things:
  -- 1. The scene is loaded in the background, and this makes sure it's
  --    done before we continue
  -- 2. It makes it easier to see what's going on
  Ips.fitScene()
  return loaded
end

local function clearScene()
  local roots = {
    Ips.getActiveObjectsRoot(),
    Ips.getGeometryRoot(),
    Ips.getMeasuresRoot(),
    Ips.getMechanismRoot(),
  }
  for _, root in pairs(roots) do
    while root:getNumChildren() > 0 do
      local child = root:getLastChild()
      Ips.deleteTreeObject(child)
    end
  end
end

local function createHarnessRouter(harnessSetup)
  local cableSim = CableSimulation()
  local harnessRouter = HarnessRouter()
  local treeObject = Ips.getActiveObjectsRoot()

  for _, cable in pairs(harnessSetup.cables) do
    local startNode = treeObject:findFirstMatch(cable.start_node)
    local startFrame = startNode:getFirstChild()
    local startVis = startFrame:toCableMountFrameVisualization()
    local endNode = treeObject:findFirstMatch(cable.end_node)
    local endFrame = endNode:getFirstChild()
    local endVis = endFrame:toCableMountFrameVisualization()
    local myCableType = cableSim:getComponentTemplate(cable.cable_type)
    harnessRouter:addSegmentTerminalMountFrames(startVis, endVis, myCableType)
  end

  for _, geometry in pairs(harnessSetup.geometries) do
    local envGeom = treeObject:findFirstMatch(geometry.name)
    local pref
    if geometry.preference == "Near" then
      pref = 0
    elseif geometry.preference == "Avoid" then
      pref = 1
    else
      pref = 2
    end
    harnessRouter:addEnvironmentGeometry(envGeom, geometry.clearance / 1000, pref, geometry.clipable)
  end

  harnessRouter:setMinMaxClipClipDist(harnessSetup.clip_clip_dist[1], harnessSetup.clip_clip_dist[2])
  harnessRouter:setMinMaxBranchClipDist(harnessSetup.branch_clip_dist[1], harnessSetup.branch_clip_dist[2])
  harnessRouter:setMinBoundingBox(false)
  harnessRouter:computeGridSize(0.02)
  harnessRouter:buildCostField()

  return harnessRouter
end

local function getCostField(harnessSetup)
  local harnessRouter = createHarnessRouter(harnessSetup)
  local gridSize = harnessRouter:getGridSize()
  local coords = {}
  local costs = {}
  -- Using 1-based indexing to get arrays when packing with msgpack
  for i_x = 1, gridSize[0], 1 do
    coords[i_x] = {}
    costs[i_x] = {}
    for i_y = 1, gridSize[1], 1 do
      coords[i_x][i_y] = {}
      costs[i_x][i_y] = {}
      for i_z = 1, gridSize[2], 1 do
        -- IPS uses 0-based indexing for grid nodes
        local coord = harnessRouter:getNodePosition(i_x - 1, i_y - 1, i_z - 1)
        local cost = harnessRouter:getNodeCost(i_x - 1, i_y - 1, i_z - 1)
        coords[i_x][i_y][i_z] = {coord.x, coord.y, coord.z}
        costs[i_x][i_y][i_z] = cost
      end
    end
  end
  return {coords=coords, costs=costs}
end

local function setHarnessRouterNodeCosts(harnessRouter, costsArray)
  -- Sets the costs of the harness router's nodes to the values in `costsArray`.
  -- The array should be 1-indexed, but the router's nodes are 0-indexed.
  -- i.e., `costsArray[1][1][1]` will be the cost of the router's node (0, 0, 0).
  for x, xCosts in ipairs(costsArray) do
    for y, yCosts in ipairs(xCosts) do
      for z, cost in ipairs(yCosts) do
        harnessRouter:setNodeCost(x - 1, y - 1, z - 1, cost)
      end
    end
  end
end

local function routeHarnessSolutions(harnessSetup, costs, bundlingFactor, namePrefix, solutionIdxsToCapture, smoothSolutions, buildDiscreteSolutions, buildPresmoothSolutions, buildSmoothSolutions)
  local harnessRouter = createHarnessRouter(harnessSetup)
  setHarnessRouterNodeCosts(harnessRouter, costs)
  harnessRouter:setObjectiveWeights(1, bundlingFactor, bundlingFactor)
  harnessRouter:routeHarness()

  local numSolutions = harnessRouter:getNumSolutions()
  local solutions = {}

  -- To be able to build the smooth segments, this step needs to be run first
  if smoothSolutions or buildSmoothSolutions then
    harnessRouter:smoothHarness()
  end

  if #solutionIdxsToCapture == 0 then
    -- If no solutions are specified, capture all of them
    solutionIdxsToCapture = range(0, numSolutions - 1)
  end

  for _, solIdx in pairs(solutionIdxsToCapture) do
    local solutionName = namePrefix .. "_" .. solIdx

    local segments = {}
    local numSegments = harnessRouter:getNumBundleSegments(solIdx)
    for segIdx = 0, numSegments - 1 do
      local segment = {
        radius = harnessRouter:getSegmentRadius(solIdx, segIdx),
        cables = vectorToTable(harnessRouter:getCablesInSegment(solIdx, segIdx)),
        discreteNodes = vectorToTable(harnessRouter:getDiscreteSegment(solIdx, segIdx, false)),
        presmoothCoords = vectorToTable(harnessRouter:getPresmoothSegment(solIdx, segIdx, false)),
        smoothCoords = nil,
        clipPositions = nil,
      }

      if smoothSolutions then
        -- These are only available if we have run the smoothing step
        segment.smoothCoords = vectorToTable(harnessRouter:getSmoothSegment(solIdx, segIdx, false))
        segment.clipPositions = vectorToTable(harnessRouter:getClipPositions(solIdx, segIdx))
      end

      segments[segIdx + 1] = segment
    end

    if buildDiscreteSolutions then
      local builtDiscreteSegmentsTreeVector = harnessRouter:buildDiscreteSegments(solIdx)
      local builtDiscreteSolution = builtDiscreteSegmentsTreeVector[0]:getParent()
      builtDiscreteSolution:setLabel(solutionName .. " (discrete)")
    end

    if buildPresmoothSolutions then
      local builtPresmoothSegmentsTreeVector = harnessRouter:buildPresmoothSegments(solIdx)
      local builtPresmoothSolution = builtPresmoothSegmentsTreeVector[0]:getParent()
      builtPresmoothSolution:setLabel(solutionName .. " (presmooth)")
    end

    if buildSmoothSolutions then
      local builtSmoothSegmentsTreeVector = harnessRouter:buildSmoothSegments(solIdx, true)
      local builtSmoothSolution = builtSmoothSegmentsTreeVector[0]:getParent()
      builtSmoothSolution:setLabel(solutionName .. " (smooth)")
    end

    -- Gather the solution data
    -- Note that we index by 1 here for packing reasons
    solutions[solIdx + 1] = {
      name = solutionName,
      segments = segments,
      estimatedNumClips = harnessRouter:estimateNumClips(solIdx),
      numBranchPoints = harnessRouter:getNumBranchPoints(solIdx),
      objectiveWeightBundling = harnessRouter:getObjectiveWeightBundling(solIdx),
      solutionObjectiveBundling = harnessRouter:getSolutionObjectiveBundling(solIdx),
      solutionObjectiveLength = harnessRouter:getSolutionObjectiveLength(solIdx),
    }
  end

  return solutions
end

local function coordDistancesToGeo(coords, geoNames)
  local treeObject = Ips.getActiveObjectsRoot()
  local prim = PrimitiveShape.createSphere(0.001, 6, 6)
  local rigid_prim = Ips.createRigidBodyObject(prim)
  local primTree = TreeObjectVector()
  primTree:insert(0, rigid_prim)

  local partsTree = TreeObjectVector()
  for _, part in pairs(geoNames) do
    partsTree:insert(0, treeObject:findFirstMatch(part))
  end

  local r = Rot3(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0))
  local measure = DistanceMeasure(1, partsTree, primTree)

  local distances = {}
  for coordIdx, coord in pairs(coords) do
    local trans = Transf3(r, Vector3d(coord[1], coord[2], coord[3]))
    rigid_prim:setFrameInWorld(trans)
    distances[coordIdx] = measure:getDistance()
  end

  Ips.deleteTreeObject(measure)
  Ips.deleteTreeObject(rigid_prim)

  return distances
end

local function evalErgo(geoNames, manikinFamilyName, coords)
  local function copy_to_static_geometry(part_table)
    for _, part_name in pairs(part_table) do
      local localtreeobject = Ips.getActiveObjectsRoot()
      local localobject = localtreeobject:findFirstExactMatch(part_name)
      local localrigidObject = localobject:toRigidBodyObject()
      localrigidObject:setLocked(false)
      local localnum_of_childs = localrigidObject:getNumChildren()
      local localgeometryRoot = Ips.getGeometryRoot()
      local localpositionedObject
      local localtoCopy
      for i = 1, localnum_of_childs do
        if i == 1 then
          localpositionedObject = localrigidObject:getFirstChild()
          localtoCopy = localpositionedObject:isPositionedTreeObject()
        else
          localpositionedObject = localpositionedObject:getNextSibling()
          localtoCopy = localpositionedObject:isPositionedTreeObject()
        end
        if localtoCopy then
          Ips.copyTreeObject(localpositionedObject, localgeometryRoot)
        end
      end
      localrigidObject:setLocked(true)
    end
  end

  copy_to_static_geometry(geoNames)

  local treeobject = Ips.getActiveObjectsRoot()

  local gp = treeobject:findFirstExactMatch("gp1")
  local gp1=gp:toGripPointVisualization()
  local gp2=gp1:getGripPoint()
  local family = treeobject:findFirstExactMatch("Family 1")
  local f1=family:toManikinFamilyVisualization()
  local f2=f1:getManikinFamily()
  f2:enableCollisionAvoidance()
  local representativeManikin = f2:getRepresentative()

  local measureTree = Ips.getMeasuresRoot()
  local measure = measureTree:findFirstExactMatch("measure")
  local measure_object = measure:toMeasure()
  local gp_geo = treeobject:findFirstExactMatch("gripGeo")
  local gp_geo1 = gp_geo:toPositionedTreeObject()

  local ergoStandards = vectorToTable(f2:getErgoStandards())
  local outputTable = {
    ergoStandards = ergoStandards,
    ergoValues = {},
    gripDiffs = {},
  }
  for coordIdx, coord in pairs(coords) do
    gp_geo1:transform(coord[1], coord[2], coord[3], 0, 0, 0)
    Ips.moveTreeObject(gp, family)
    f2:posePredict(10)
    -- updateScreen needed for measure to work
    Ips.updateScreen()
    local dist = measure_object:getValue()

    local coordErgoValues = {}
    for ergoStandardIdx, ergoStandard in pairs(ergoStandards) do
      local ergoValue = f2:evaluateStaticErgo(ergoStandard, representativeManikin)
      coordErgoValues[ergoStandardIdx] = ergoValue
    end
    outputTable.ergoValues[coordIdx] = coordErgoValues
    outputTable.gripDiffs[coordIdx] = dist
  end
  return outputTable
end

module.type = _type
module.pack = _pack
module.unpack = _unpack
module.vectorToTable = vectorToTable
module.range = range

module.loadAndFitScene = loadAndFitScene
module.clearScene = clearScene
module.getCostField = getCostField
module.setHarnessRouterNodeCosts = setHarnessRouterNodeCosts
module.routeHarnessSolutions = routeHarnessSolutions
module.coordDistancesToGeo = coordDistancesToGeo
module.evalErgo = evalErgo

module.base64 = base64
module.inspect = inspect
module.msgpack = msgpack

return module
