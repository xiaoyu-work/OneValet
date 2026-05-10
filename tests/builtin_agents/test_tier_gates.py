from koa.builtin_agents.email.tools import (
    archive_emails,
    delete_emails,
    mark_as_read,
    reply_email,
    search_emails,
    send_email,
)
from koa.builtin_agents.shipment.agent import ShippingAgent
from koa.builtin_agents.shipment.tools import track_shipment
from koa.builtin_agents.trip_planner.travel_tools import search_flights, search_hotels
from koa.agents.decorator import get_agent_metadata


def test_pro_only_email_write_tools_are_tier_gated():
    assert search_emails.enabled_tiers is None
    for tool in (send_email, reply_email, delete_emails, archive_emails, mark_as_read):
        assert tool.enabled_tiers == ["pro"]


def test_pro_only_travel_and_shipment_tools_are_tier_gated():
    assert search_flights.enabled_tiers == ["pro"]
    assert search_hotels.enabled_tiers == ["pro"]
    assert track_shipment.enabled_tiers == ["pro"]


def test_shipping_agent_requires_pro_tier():
    metadata = get_agent_metadata(ShippingAgent)

    assert metadata is not None
    assert metadata.extra["required_tier"] == "pro"
