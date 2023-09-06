def load_scene(ips_instance, file_path):
    escaped_string = file_path.encode('unicode_escape').decode() #.encode('unicode_escape').decode()
    command ="""
    local IPSFile = '""" + escaped_string + """'
    Ips.loadScene(IPSFile)
    """
    ips_instance.call(command)


def get_geometries(ips_instance):
    command = """
    local treeObject = Ips.getActiveObjectsRoot()
    nodes = "All"
    local object = treeObject:getFirstChild();
    local numbOfGeoemtries = treeObject:getNumChildren();
    type = object:getType()
    if(type=="RigidBodyObject")
    then
        nodes = nodes .. "," .. object:getLabel()
    end
    for i = 2,numbOfGeoemtries,1 
    do 
        local objectTemp = object:getNextSibling();
        object = objectTemp;
        type = object:getType()
        if(type=="RigidBodyObject")
        then
            nodes = nodes .. "," .. object:getLabel()
        else
            local numbOfchilds = object:getNumChildren();
            local objectobject = object:getFirstChild();
            type = objectobject:getType()
            if(type=="RigidBodyObject")
            then
                nodes = nodes .. "," .. objectobject:getLabel()
            end
            for ii = 2,numbOfchilds,1 
            do 
                local objectobjectTemp = objectobject:getNextSibling();
                objectobject = objectobjectTemp;
                type = objectobject:getType()
                name = objectobject:getLabel()
                if(type=="RigidBodyObject")
                then
                    nodes = nodes .. "," .. objectobject:getLabel()
                end
            end
        end
    end
    return nodes 
    """
    geometry_string = ips_instance.call(command)
    geometry_string = geometry_string.decode('utf-8').strip().replace('"', '')
    geometries = geometry_string.split(',')
    geometries.pop(0)
    return geometries

def get_nodes(ips_instance):
    command = """
    local treeObject = Ips.getActiveObjectsRoot()
    nodes = "All nodes"
    local object = treeObject:getFirstChild();
    local numbOfGeoemtries = treeObject:getNumChildren();
    type = object:getType()
    if(type=="Node")
    then
        nodes = nodes .. "," .. object:getLabel()
    end
    for i = 2,numbOfGeoemtries,1 
    do 
        local objectTemp = object:getNextSibling();
        object = objectTemp;
        type = object:getType()
        if(type=="Node")
        then
            nodes = nodes .. "," .. object:getLabel()
        else
            local numbOfchilds = object:getNumChildren();
            local objectobject = object:getFirstChild();
            type = objectobject:getType()
            if(type=="Node")
            then
                nodes = nodes .. "," .. objectobject:getLabel()
            end
            for ii = 2,numbOfchilds,1 
            do 
                local objectobjectTemp = objectobject:getNextSibling();
                objectobject = objectobjectTemp;
                type = objectobject:getType()
                name = objectobject:getLabel()
                if(type=="Node")
                then
                    nodes = nodes .. "," .. objectobject:getLabel()
                end
            end
        end
    end
    return nodes 
    """
    nodes_string = ips_instance.call(command)
    nodes_string = nodes_string.decode('utf-8').strip().replace('"', '')
    nodes = nodes_string.split(',')
    nodes.pop(0)
    return nodes
