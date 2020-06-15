using AWSCore, AWSSDK, Dates, Sockets

function spot_wrangle(no_instances::Integer, spot_price::AbstractFloat, sgrp_name::String, sgrp_desc::String, skeys::String, zone::String, ami::String, instance_type::String)
    #ssh permissions
    WANip=chomp(read(pipeline(`dig +short myip.opendns.com @resolver1.opendns.com`), String))
    permissions = [
        [
            "FromPort" => 22,
            "IpProtocol" => "tcp",
            "IpRanges" => [
                [
                    "CidrIp" => WANip*"/32",
                    "Description" => "SSH access"
                ]
            ],
            "ToPort" => 22
        ]
    ]

    @info "Configuring AWS credentials..."
    aws=aws_config()

    @info "Getting security group info..."
    sgrp=Dict()

    try
        sgrp=AWSSDK.EC2.create_security_group(aws,GroupDescription=sgrp_desc,GroupName=sgrp_name)
        inret=AWSSDK.EC2.authorize_security_group_ingress(aws, GroupId=sgrp["groupId"], IpPermissions=permissions)
        outret=AWSSDK.EC2.authorize_security_group_egress(aws, GroupId=sgrp["groupId"], IpPermissions=permissions)
    catch error
        if error.code == "InvalidGroup.Duplicate"
            sgrp=AWSSDK.EC2.describe_security_groups(aws,GroupName=sgrp_name)["securityGroupInfo"]["item"]
        else
            throw(error)
        end
    end

    #Launch specs
    LaunchSpec = [
        "ImageId" => ami,
        "InstanceType" => instance_type,
        "SecurityGroupId" => [
            sgrp["groupId"]
        ],
        "Placement" => [
            "AvailabilityZone" => zone
        ],
        "KeyName" => skeys
    ]

    @info "Requesting spot instances..."

    spot_reqs=AWSSDK.EC2.request_spot_instances(aws, LaunchSpecification=LaunchSpec, InstanceCount=no_instances, SpotPrice=spot_price, Type="one-time")

    instance_public_ips=Vector()

    @info "Waiting for instances to become active..."

    if no_instances==1
        spot_req_Id=spot_reqs["spotInstanceRequestSet"]["item"]["spotInstanceRequestId"]
        active=false
        while !active
            AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["state"] == "active" ? active=true : sleep(10)
        end
        instance_Id=AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["instanceId"]
        address=AWSSDK.EC2.describe_instances(aws, InstanceId=instance_Id)["reservationSet"]["item"]["instancesSet"]["item"]["ipAddress"]
        push!(instance_public_ips,address)
    else
        for spot_req in spot_reqs["spotInstanceRequestSet"]["item"]
            spot_req_Id = spot_req["spotInstanceRequestId"]
            active=false
            while !active
                AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["state"] == "active" ? active=true : sleep(10)
            end
            instance_Id = AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["instanceId"]
            address=AWSSDK.EC2.describe_instances(aws, InstanceId=instance_Id)["reservationSet"]["item"]["instancesSet"]["item"]["ipAddress"]
            push!(instance_public_ips,address)
        end
    end

    return instance_public_ips
end

function get_cheapest_zone(instance_type::String)
    arguments=[
        "ProductDescription"=>["Linux/UNIX"]
        "InstanceType"=>[instance_type]
        "StartTime"=>Dates.now(Dates.UTC)
    ]
    history=AWSSDK.EC2.describe_spot_price_history(aws_config(), arguments)

    price_dict=Dict{String,Float64}()
    for record in history["spotPriceHistorySet"]["item"]
        price_dict[record["availabilityZone"]]=parse(Float64, record["spotPrice"])
    end

    return sort(collect(price_dict), by=x->x[2])[1]
end
