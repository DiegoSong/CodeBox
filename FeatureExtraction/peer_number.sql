select distinct *
from `dev_ff_moonlight`.`moonlight_overdue` a
left join 
(
select * from
    (
    select mobile, 
        sum(case when calls_items_peer_number in ('10086','10000','10010','10085','10001','12580','1008611','1001011','0281008611') then 1 else 0 end) operator,   
        sum(case when calls_items_peer_number='12599' then 1 else 0 end) cm_message, --'移动语音留言'   
        sum(case when calls_items_peer_number in ('95511','95510','95518','95542','95500','95522','95519','95589','95590',
                                        '95585','95505','95550','95567','95556','95308','95300') then 1 else 0 end) insurance_cs, -- '保险公司客服'   
        sum(case when calls_items_peer_number in ('02895510','01095510','02795522','075595511', '4008688888','4008803633','4008777777') then 1 else 0 end) insurance_mk, -- '保险公司营销'  
        sum(case when length(calls_items_peer_number)=5 and calls_items_peer_number like '955%'
                and calls_items_peer_number in ('95588','95599','95566','95533','95559','95528','95561','95595','95501','95568','95558','95555','95508',
                '95577','96169','11158','962888','95516','95566','95588','95533','95599','95555','95559','95558','95568',
                '95561','95508','95528','95595','95577','95568','95501','4006699999','8008303811','8008302880','8008208088',
                '8008301880','8008304888','8008308008') then 1 else 0 end) bank, --'银行'  
        sum(case when calls_items_peer_number in ('95338','95311','95320','95543','95554','95546') then 1 else 0 end) express, --'快递'  
        sum(case when calls_items_peer_number in ('11183','11185') then 1 else 0 end) ems, --'邮政'
        sum(case when calls_items_peer_number in ('95066','4000000999','4000000666') then 1 else 0 end) taxi, --'网约车' 
        sum(case when calls_items_peer_number in ('95079','10011','4008810000','10050','96196','95079') then 1 else 0 end) broadband, --'宽带' 
        sum(case when calls_items_peer_number in ('95079','95010') then 1 else 0 end) travel, --'旅行'  
        sum(case when calls_items_peer_number in ('95353') then 1 else 0 end) logistics, -- '物流'  
        sum(case when calls_items_peer_number in ('95118','95177','4006065500','4006999999') then  1 else 0 end) shopping, --'家电购物'  
        sum(case when calls_items_peer_number in ('4001609511') then  1 else 0 end) collect,-- '催收'
        sum(case when calls_items_peer_number in ('10100000','4000271262','4008580580','4000368876','4000609191','4008108818','4001848888',
                                        '4009008880','4006808888','4006363636','4009987101','4001809860','4000100788',
                                        '4001868888','4000271268','4006993906','4006551250','02110100000','95183','95055','95137') then  1 else 0 end) loan, --'贷款电话'  
        sum(case when calls_items_peer_number in ('02195508','02795559','01095580','02895568') then 1 else 0 end) credit_mk, --'信用卡营销'
        sum(case when calls_items_peer_number in ('4008895558','4008009888','4006695599','4006695566',
                                        '4006695568','4008205555','4008880288','4008208788','4006695577','4007888888','4008895580',
                                        '4008200588','4008111333','4008108008','4006595555','4008893888','4001000000','4006095558',
                                        '4007207888','4000095555','4001895555') then 1 else 0 end) credit_cs, --'信用卡客服'
        sum(case when calls_items_peer_number in ('4006668800','4001005678','4008308300','4006789688','4001666888') then 1 else 0 end) mobile_cs, --'手机'  
        sum(case when calls_items_peer_number in ('075583765566','02083918160','02258329888') then 1 else 0 end) games, --'游戏'  
        sum(case when calls_items_peer_number in ('95188','95017','95016') then 1 else 0 end) pay, --'支付'  
        sum(case when substr(calls_items_peer_number,1,3) in ('139','138','186','135','137','136','158','159','150','189','151','152','187','133',
                                                    '182','180','183','185','134','188','153','181','130','131','156','177','155','132',
                                                    '157','176','173','178','184','199','175','147','170','171','186') then 1 else 0 end) mobile_phone--'移动电话'  
        from gdmjsj_sda.sda01_moxie_mobile_carrier_data_calls_items_flat 
        group by mobile
        ) a
        join operator_features_new b
        on a.mobile = b.phone
    ) ab
on a.phone=ab.phone
where ab.phone is not null
and a.plan_repay_date is not null
