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

local function log(msg)
  -- Logs a message to the IPS log
  print("Autopack: " .. msg)
end

local function pause(msg)
  -- Pauses the script until the user presses enter.
  local answer = Ips.question((msg or "") .. "\n\nContinue?")
  if answer == false then
    error("Script aborted")
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

local function treeObjChildren(treeObj)
  local children = {}
  local numChildren = treeObj:getNumChildren()
  for i = 1, numChildren do
    if i == 1 then
      children[i] = treeObj:getFirstChild()
    else
      children[i] = children[i - 1]:getNextSibling()
    end
  end
  return children
end

local function loadAndFitScene(scenePath)
  log("Loading scene " .. scenePath)
  local loaded = Ips.loadScene(scenePath)
  -- Fitting the scene helps with two things:
  -- 1. The scene is loaded in the background, and this makes sure it's
  --    done before we continue
  -- 2. It makes it easier to see what's going on
  Ips.fitScene()
  return loaded
end

local function clearScene()
  -- Clears the scene of all active objects, static geometry, measures,
  -- and mechanisms
  local roots = {
    -- Start with processes, as they may have dependencies that are
    -- active objects
    Ips.getProcessRoot(),
    Ips.getActiveObjectsRoot(),
    Ips.getGeometryRoot(),
    Ips.getMeasuresRoot(),
    Ips.getMechanismRoot(),
    Ips.getSimulationsRoot(),
  }
  for _, root in pairs(roots) do
    while root:getNumChildren() > 0 do
      local child = root:getLastChild()
      Ips.deleteTreeObject(child)
    end
  end
  log("Scene cleared")
end

local function getOrCreateActiveGroup(groupName, parent)
  parent = parent or Ips.getActiveObjectsRoot()
  local activeGroup = parent:findFirstExactMatch(groupName)
  if not activeGroup then
    activeGroup = Ips.createAssembly(groupName)
    Ips.moveTreeObject(activeGroup, parent)
  end
  return activeGroup
end

local function getOrCreateGeometryGroup(groupName, parent)
  parent = parent or Ips.getGeometryRoot()
  local geoGroup = parent:findFirstExactMatch(groupName)
  if not geoGroup then
    geoGroup = Ips.createGeometryGroup(parent)
    geoGroup:setLabel(groupName)
  end
  return geoGroup
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
        coords[i_x][i_y][i_z] = { coord.x, coord.y, coord.z }
        costs[i_x][i_y][i_z] = cost
      end
    end
  end
  return { coords = coords, costs = costs }
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

local function routeHarnesses(
  harnessSetup,
  costs,
  bundlingFactor,
  namePrefix,
  solutionIdxsToCapture,
  smoothSolutions,
  buildDiscreteSolutions,
  buildPresmoothSolutions,
  buildSmoothSolutions,
  buildCableSimulations
)
  local harnessGroupName = "Autopack harnesses"
  local harnessGeoGroup = getOrCreateGeometryGroup(harnessGroupName)
  local harnessActiveGroup = getOrCreateActiveGroup(harnessGroupName)
  local harnessRouter = createHarnessRouter(harnessSetup)
  setHarnessRouterNodeCosts(harnessRouter, costs)
  harnessRouter:setObjectiveWeights(1, bundlingFactor, bundlingFactor)
  harnessRouter:routeHarness()

  local numSolutions = harnessRouter:getNumSolutions()
  local solutions = {}
  local smoothSolutionsAvailable = false

  -- To be able to build the smooth segments or cable simulations, this
  -- step needs to be run first
  if smoothSolutions or buildSmoothSolutions or buildCableSimulations then
    harnessRouter:smoothHarness()
    smoothSolutionsAvailable = true
  end

  if #solutionIdxsToCapture == 0 then
    -- If no solutions are specified, capture all of them
    solutionIdxsToCapture = range(0, numSolutions - 1)
  end

  for _, solIdx in pairs(solutionIdxsToCapture) do
    local solutionName = namePrefix .. "." .. solIdx
    log("Capturing solution " .. solutionName)

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

      if smoothSolutionsAvailable then
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
      Ips.moveTreeObject(builtDiscreteSolution, harnessGeoGroup)
    end

    if buildPresmoothSolutions then
      local builtPresmoothSegmentsTreeVector = harnessRouter:buildPresmoothSegments(solIdx)
      local builtPresmoothSolution = builtPresmoothSegmentsTreeVector[0]:getParent()
      builtPresmoothSolution:setLabel(solutionName .. " (presmooth)")
      Ips.moveTreeObject(builtPresmoothSolution, harnessGeoGroup)
    end

    if buildSmoothSolutions then
      local builtSmoothSegmentsTreeVector = harnessRouter:buildSmoothSegments(solIdx, true)
      local builtSmoothSolution = builtSmoothSegmentsTreeVector[0]:getParent()
      builtSmoothSolution:setLabel(solutionName .. " (smooth)")
      Ips.moveTreeObject(builtSmoothSolution, harnessGeoGroup)
    end

    if buildCableSimulations then
      local builtCableSimulation = harnessRouter:buildSimulationObject(solIdx, true)
      if builtCableSimulation:hasExpired() then
        log("Failed to build simulation object for solution " .. solutionName)
        getOrCreateActiveGroup(solutionName .. " (failed)", harnessActiveGroup)
      else
        builtCableSimulation:setLabel(solutionName)
        Ips.moveTreeObject(builtCableSimulation, harnessActiveGroup)
      end
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

local function coordDistancesToGeo(coords, geoNames, includeStaticGeo)
  local activeObjsRoot = Ips.getActiveObjectsRoot()
  local staticGeoRoot = Ips.getGeometryRoot()
  local dot = Ips.createRigidBodyObject(PrimitiveShape.createSphere(0.0001, 6, 6))

  local dotVector = TreeObjectVector()
  dotVector:push_back(dot)

  local partVector = TreeObjectVector()
  for _, part in pairs(geoNames) do
    partVector:push_back(activeObjsRoot:findFirstExactMatch(part))
  end
  if includeStaticGeo then
    partVector:push_back(staticGeoRoot)
  end

  local measure = DistanceMeasure(DistanceMeasure.MODE_1_VS_2, partVector, dotVector)

  local distances = {}
  for coordIdx, coord in pairs(coords) do
    dot:transform(coord[1], coord[2], coord[3], 0, 0, 0)
    distances[coordIdx] = measure:getDistance()
  end

  Ips.deleteTreeObject(measure)
  Ips.deleteTreeObject(dot)

  return distances
end

local function copyRigidBodyGeometry(rigidBody, destTreeObj)
  rigidBody:setLocked(false)
  for _, child in pairs(treeObjChildren(rigidBody)) do
    if child:isPositionedTreeObject() and not child:isFrame() then
      Ips.copyTreeObject(child, destTreeObj)
    end
  end
  rigidBody:setLocked(true)
end

local function moveGripPoint(gripPointViz, translationVector)
  -- local r = Rot3(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0))
  -- local transf = Transf3(r, translationVector)
  -- Does not work
  -- gripPoint:setTarget(transf)
  -- gripPoint:getVisualization():setTWorld(transf)
  -- Almost works
  -- gripPoint:getVisualization():setTControl(transf)
  -- Works
  return gripPointViz:transform(translationVector.x, translationVector.y, translationVector.z, 0, 0, 0)
end

local function getManikinCtrlPoint(familyViz, ctrlPointName)
  return familyViz:findFirstExactMatch(ctrlPointName):toControlPointVisualization():getControlPoint()
end

local function copyToStaticGeometry(activeObjNames)
  -- Copies the rigid bodies with the given names to the static geometry
  -- root
  local activeObjRoot = Ips.getActiveObjectsRoot()
  local destTreeObj = getOrCreateGeometryGroup("Autopack copied geometry")
  for _, activeObjName in pairs(activeObjNames) do
    local rigidBody = activeObjRoot:findFirstExactMatch(activeObjName):toRigidBodyObject()
    copyRigidBodyGeometry(rigidBody, destTreeObj)
  end
  return destTreeObj
end

local function getAllManikinFamilies()
  local msc = ManikinSimulationController()
  -- Manikin family IDs are UUIDs, not related to names or indices
  local manikinFamilyIds = vectorToTable(msc:getManikinFamilyIDs())
  local manikinFamilyNames = {}
  for _, manikinFamilyId in pairs(manikinFamilyIds) do
    manikinFamilyNames[#manikinFamilyNames + 1] = {
      id = manikinFamilyId,
      name = msc:getManikinFamily(manikinFamilyId):getVisualization():getLabel(),
    }
  end

  return manikinFamilyNames
end

local function evalErgo(geoNames, manikinFamilyId, coords, enableRbpp, updateScreen, keepGenObj)
  local copiedGeoGroup = copyToStaticGeometry(geoNames)

  local msc = ManikinSimulationController()
  local family = msc:getManikinFamily(manikinFamilyId)
  local familyViz = family:getVisualization()
  local ergoStandards = vectorToTable(family:getErgoStandards())
  family:enableCollisionAvoidance()

  local gripPoint = msc:createGripPoint()
  local gripPointViz = gripPoint:getVisualization()
  -- genObjsRoot:insert(0, gripPoint)
  gripPoint:setGripConfiguration("Tip Pinch")
  gripPoint:setSymmetricRotationTolerances(math.huge, math.huge, math.huge)
  gripPoint:setSymmetricTranslationTolerances(0.005, 0.005, 0.005)

  local opSequence = OperationSequence()
  opSequence:setLabel("Autopack ergo evaluation")
  local familyActor = opSequence:addFamilyActor(familyViz)
  familyActor:setCurrentStateAsStart()
  -- Add a pause action so we get a time where the manikin is steadily
  -- in its start state
  local pauseAction = opSequence:createManikinWaitAction(familyActor, 1e-6)
  local graspAction = opSequence:createManikinGraspAction(familyActor, gripPointViz)
  if enableRbpp then
    graspAction:enableRigidBodyPathPlanning()
  else
    graspAction:disableRigidBodyPathPlanning()
  end
  -- Add a release action as it seems to help with resetting the
  -- manikins properly
  local releaseAction = opSequence:createManikinReleaseAction(familyActor, gripPointViz)
  releaseAction:maintainCurrentPosture()

  local outputTable = {
    ergoStandards = ergoStandards,
    ergoValues = {},
    gripDiffs = {},
    errorMsgs = {},
  }
  local replay -- Declared here to enable resetting after the loop
  local pauseActionEndTime
  for coordIdx, coord in pairs(coords) do
    moveGripPoint(gripPointViz, Vector3d(coord[1], coord[2], coord[3]))
    replay = opSequence:executeSequence()
    pauseActionEndTime = replay:getActionEndTime(pauseAction)
    local graspActionEndTime = replay:getActionEndTime(graspAction)

    if updateScreen then
      Ips.updateScreen()
    end

    -- The control point gets deleted (???) after manipulating the manikin, so we must get it when we need it
    local handRightTransl = getManikinCtrlPoint(familyViz, "Right Hand"):getTarget().t
    local gripPointTransl = gripPoint:getTarget().t
    local dist = handRightTransl:distance(gripPointTransl)

    local coordErgoValues = {}
    for ergoStandardIdx, ergoStandard in pairs(ergoStandards) do
      local ergoValues = vectorToTable(replay:computeErgonomicScore(ergoStandard, graspActionEndTime, graspActionEndTime))
      coordErgoValues[ergoStandardIdx] = ergoValues
    end
    outputTable.ergoValues[coordIdx] = coordErgoValues
    outputTable.gripDiffs[coordIdx] = dist
    outputTable.errorMsgs[coordIdx] = replay:getReplayErrorMessage(graspAction)
    log("Ergo evaluation: " .. coordIdx .. "/" .. #coords .. " done")
  end
  -- Make sure that the replay is rewinded, to set the manikin back to
  -- its start state. Rewinding to 0.0 does not properly reset the start
  -- state.
  replay:setTime(pauseActionEndTime)

  if not keepGenObj then
    Ips.deleteTreeObject(opSequence)
    Ips.deleteTreeObject(gripPointViz)
    Ips.deleteTreeObject(copiedGeoGroup)
  end

  return outputTable
end

local function createColoredPointCloud(points, treeParent, treeObjName, removeExisting)
  -- `points` is an array of arrays, where each sub-array is a point, described by 6 numbers:
  -- x, y, z, r, g, b
  local staticGeoRoot = Ips.getGeometryRoot()
  if not treeParent then
    treeParent = staticGeoRoot
  end
  if removeExisting then
    local existingTreeObj = treeParent:findFirstExactMatch(treeObjName)
    if existingTreeObj then
      Ips.deleteTreeObject(existingTreeObj)
    end
  end
  local builder = GeometryBuilder()
  for pointIdx, point in pairs(points) do
    builder:pushVertex(point[1], point[2], point[3])
    builder:pushColor(point[4], point[5], point[6])
  end

  builder:buildPoints()
  local treeObj = staticGeoRoot:getLastChild()
  Ips.moveTreeObject(treeObj, treeParent)
  treeObj:setLabel(treeObjName)

  return treeObj
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
module.routeHarnesses = routeHarnesses
module.coordDistancesToGeo = coordDistancesToGeo
module.getAllManikinFamilies = getAllManikinFamilies
module.evalErgo = evalErgo
module.createColoredPointCloud = createColoredPointCloud

module.base64 = base64
module.inspect = inspect
module.msgpack = msgpack

return module
